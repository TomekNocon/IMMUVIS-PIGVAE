from typing import Any

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.data.components.graphs_datamodules import DenseGraphBatch
from src.models.components.embeddings import PositionalEncoding

# from src.models.components.custom_graph_transformer import Transformer
from src.models.components.llama_graph_transformer import Transformer
from src.models.components.rotary_embedding import LLamaRotaryEmbedding
from src.models.components.spectral_embeddings import SklearnSpectralEmbedding


class NodeStatsProjection(nn.Module):
    """Residual correction to CLS z from pooled node statistics (mean, var, max).

    Zero-initialized so the correction starts at 0 — training begins identical
    to the baseline and the model learns when/how much to use the stats.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(3 * hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        # node_features: [B, N, D]
        mean = node_features.mean(dim=1)                      # [B, D]
        var  = node_features.var(dim=1, unbiased=False)       # [B, D]
        max_ = node_features.max(dim=1).values                # [B, D]
        stats = torch.cat([mean, var, max_], dim=-1)          # [B, 3D]
        return self.proj(stats)                               # [B, D]


class StructuralCorrection(nn.Module):
    """Residual correction to CLS graph_emb from grid-topology structural features.

    Supports three composable components (Ideas 3, 4, 6 from latent_enrichment_ideas.md):
      use_hadamard  (Idea 6): mean of h_i ⊙ h_j over grid edges — zero-param, fast
      use_mlp_edges (Idea 3): mean of MLP(concat(h_i, h_j)) — learns cross-dim interactions
      use_spectrum  (Idea 4): top-k eigenvalues of content-weighted Laplacian — global structure

    All components use a precomputed edge_index buffer (fixed 6×6 grid topology).
    DenseGraphBatch.edge_features is never read. The final projection is zero-initialized
    so training begins identical to the baseline regardless of which flags are on.
    """

    def __init__(
        self,
        hidden_dim: int,
        grid_size: int = 6,
        use_hadamard: bool = True,
        use_mlp_edges: bool = False,
        use_spectrum: bool = False,
        n_spectral: int = 8,
    ):
        super().__init__()
        self.use_hadamard = use_hadamard
        self.use_mlp_edges = use_mlp_edges
        self.use_spectrum = use_spectrum
        self.n_spectral = n_spectral

        G = nx.grid_2d_graph(grid_size, grid_size)
        node_to_idx = {n: i for i, n in enumerate(G.nodes())}
        edge_index = torch.tensor(
            [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()], dtype=torch.long
        )
        self.register_buffer("edge_index", edge_index)  # [E, 2]

        if use_spectrum:
            A = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32)
            self.register_buffer("A_grid", A)  # [N, N]

        if use_mlp_edges:
            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.GELU(),
            )

        in_dim = (hidden_dim if use_hadamard else 0) + \
                 (hidden_dim if use_mlp_edges else 0) + \
                 (n_spectral if use_spectrum else 0)
        assert in_dim > 0, "Enable at least one of use_hadamard, use_mlp_edges, use_spectrum"

        self.proj = nn.Linear(in_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        # node_features: [B, N, D]
        h_i = node_features[:, self.edge_index[:, 0], :]  # [B, E, D]
        h_j = node_features[:, self.edge_index[:, 1], :]  # [B, E, D]

        parts = []

        if self.use_hadamard:
            parts.append((h_i * h_j).mean(dim=1))          # [B, D]

        if self.use_mlp_edges:
            parts.append(
                self.edge_mlp(torch.cat([h_i, h_j], dim=-1)).mean(dim=1)  # [B, D]
            )

        if self.use_spectrum:
            h_norm = F.normalize(node_features, dim=-1)
            sim = torch.bmm(h_norm, h_norm.transpose(1, 2))           # [B, N, N]
            W = F.relu(sim) * self.A_grid.unsqueeze(0)                 # [B, N, N]
            D_w = torch.diag_embed(W.sum(dim=-1))
            L_w = D_w - W
            N = node_features.shape[1]
            L_w = L_w + 1e-4 * torch.eye(N, device=L_w.device)
            eigenvalues = torch.linalg.eigvalsh(L_w)                   # [B, N]
            parts.append(eigenvalues[:, :self.n_spectral])             # [B, n_spectral]

        return self.proj(torch.cat(parts, dim=-1))                     # [B, hidden_dim]


class PMAReadout(nn.Module):
    """K-seed cross-attention pooling (Pool by Multihead Attention, Lee et al. 2019).

    Drop-in replacement for the CLS token readout:
      k=1  → single seed, cross-attends to all nodes — equivalent to CLS but computed
              after (not inside) the transformer, so the transformer sees only content nodes.
      k>1  → K independent summaries, concatenated and projected to hidden_dim.

    Set use_pma=False in GraphEncoder to keep the original CLS-in-transformer approach.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, k: int = 4):
        super().__init__()
        self.k = k
        self.seeds = nn.Parameter(torch.empty(k, hidden_dim))
        nn.init.trunc_normal_(self.seeds, std=0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.proj = nn.Linear(k * hidden_dim, hidden_dim, bias=False) if k > 1 else nn.Identity()

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        # node_features: [B, N, D]
        B = node_features.shape[0]
        seeds = self.seeds.unsqueeze(0).expand(B, -1, -1)                           # [B, K, D]
        pooled, _ = self.cross_attn(query=seeds, key=node_features, value=node_features)  # [B, K, D]
        return self.proj(pooled.flatten(1))                                          # [B, D]


class NodeBottleneckEncoder(nn.Module):
    """Per-node bottleneck: [B, N, D] → [B, N, node_z_dim].

    Projects each node's encoder representation to a compact per-node latent.
    No pooling — spatial arrangement is preserved across all N positions.

    In VAE mode each node position gets its own mu/logvar. The same eps is
    shared across all augmented views of the same underlying sample so that
    the stochastic perturbation is consistent across D4 orientations.
    """

    def __init__(self, in_dim: int, node_z_dim: int, vae: bool = False, num_permutations: int = 8):
        super().__init__()
        self.vae = vae
        self.num_permutations = num_permutations
        self.proj = nn.Linear(in_dim, node_z_dim * 2 if vae else node_z_dim)
        # No LayerNorm on the latent: normalising z per node removes per-node
        # magnitude (intensity), which the decoder needs to reconstruct. Latent
        # scale is handled by the decoder's first pre-norm anyway.

    def forward(
        self, node_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        out = self.proj(node_features)
        if not self.vae:
            return out, None, None

        mu, logvar = out.chunk(2, dim=-1)           # [B, N, node_z_dim] each
        logvar = torch.clamp(logvar, -10, 10)
        std = (0.5 * logvar).exp()

        # Generate eps for base samples only, then tile across augmented views
        # so every orientation of the same patch uses the same noise.
        base_bs = node_features.shape[0] // self.num_permutations
        eps_base = torch.randn_like(std[:base_bs])  # [B/K, N, node_z_dim]
        eps = eps_base.unsqueeze(0).repeat(self.num_permutations, 1, 1, 1)
        eps = eps.view(-1, *eps_base.shape[1:])     # [B, N, node_z_dim]

        return mu + eps * std, mu, logvar


class GraphAE(torch.nn.Module):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.input_size = hparams.input_size
        self.vae = hparams.vae
        self.encoder = GraphEncoder(hparams.encoder)
        self.node_bottleneck = NodeBottleneckEncoder(
            hparams.encoder.graph_encoder_hidden_dim,
            hparams.node_z_dim,
            vae=hparams.vae,
            num_permutations=hparams.permuter.num_permutations,
        )
        self.permuter = SimplePermuter(hparams.permuter)
        self.decoder = GraphDecoder(hparams.decoder)

    def encode(
        self, graph: DenseGraphBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        graph_emb, node_features = self.encoder(
            node_features=graph.node_features,
            edge_features=graph.edge_features,
            mask=graph.mask,
        )
        z_nodes, mu, logvar = self.node_bottleneck(node_features)  # [B, N, node_z_dim]
        z_global = F.layer_norm(graph_emb, graph_emb.shape[-1:])   # [B, D] — CLS for FiLM
        return z_nodes, z_global, node_features, mu, logvar

    def decode(
        self,
        z_nodes: torch.Tensor,
        z_global: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> DenseGraphBatch:
        node_logits, edge_logits = self.decoder(z_nodes, z_global, mask)
        return DenseGraphBatch(
            node_features=node_logits,
            edge_features=edge_logits,
            mask=mask,
            properties=torch.tensor([]),
        )

    def forward(self, graph: DenseGraphBatch, training: bool, tau: float = 1.0) -> tuple:
        z_nodes, z_global, _, mu, logvar = self.encode(graph=graph)
        graph_pred = self.decode(z_nodes, z_global, graph.mask)
        graph_emb = z_nodes.mean(dim=1)  # [B, node_z_dim] — mean pool for logging
        return graph_emb, graph_pred, None, None, mu, logvar


class GraphEncoder(torch.nn.Module):
    def __init__(self, hparams: DictConfig):
        super().__init__()

        self.use_pma = getattr(hparams, "use_pma", False)
        if self.use_pma:
            self.pma = PMAReadout(
                hidden_dim=hparams.graph_encoder_hidden_dim,
                num_heads=hparams.graph_encoder_num_heads,
                dropout=hparams.dropout,
                k=getattr(hparams, "pma_k", 4),
            )
        else:
            self.summary_node = nn.Parameter(torch.randn(1, 1, hparams.graph_encoder_hidden_dim))
            nn.init.trunc_normal_(self.summary_node, std=0.02)
        if hparams.project:
            self.projection_in = nn.Linear(
                hparams.num_node_features, hparams.graph_encoder_hidden_dim
            )
        self.project = hparams.project
        self.graph_transformer = Transformer(
            hidden_dim=hparams.graph_encoder_hidden_dim,
            num_heads=hparams.graph_encoder_num_heads,
            ppf_hidden_dim=hparams.graph_encoder_ppf_hidden_dim,
            num_layers=hparams.graph_encoder_num_layers,
            dropout=hparams.dropout,
            # final_norm bounds the internal pre-norm residual stream (removing it blew up
            # encoder max_abs to ~260 in run cuogzab1). Configurable so checkpoints trained
            # without it can be loaded faithfully. Default on.
            use_final_norm=getattr(hparams, "use_final_norm", True),
            qk_norm=getattr(hparams, "qk_norm", False),
        )
        self.fc_in = nn.Linear(hparams.graph_encoder_hidden_dim, hparams.graph_encoder_hidden_dim)
        self.output_norm = nn.LayerNorm(hparams.graph_encoder_hidden_dim, elementwise_affine=False)
        self.stats_correction = NodeStatsProjection(hparams.graph_encoder_hidden_dim)
        use_hadamard  = getattr(hparams, "use_hadamard",  False)
        use_mlp_edges = getattr(hparams, "use_mlp_edges", False)
        use_spectrum  = getattr(hparams, "use_spectrum",  False)
        if use_hadamard or use_mlp_edges or use_spectrum:
            self.structural_correction = StructuralCorrection(
                hidden_dim=hparams.graph_encoder_hidden_dim,
                grid_size=hparams.grid_size,
                use_hadamard=use_hadamard,
                use_mlp_edges=use_mlp_edges,
                use_spectrum=use_spectrum,
                n_spectral=getattr(hparams, "n_spectral", 8),
            )
        else:
            self.structural_correction = None
        self.dropout = nn.Dropout(hparams.dropout)

    def add_emb_node_and_feature(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = node_features.size(0)
        cls_tokens = self.summary_node.expand(batch_size, -1, -1)
        node_features = torch.cat([cls_tokens, node_features], dim=1)  # (B, N+1, D)
        mask = F.pad(mask, (1, 0), value=1)
        return node_features, edge_features, mask

    def init_message_matrix(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_features, edge_features, mask = self.add_emb_node_and_feature(
            node_features, edge_features, mask
        )
        x = self.dropout(F.silu(self.fc_in(node_features))) # altenatives are silu - but works worse with negative values or no activation
        return x, mask

    def read_out_message_matrix(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        node_features = x
        graph_emb, node_features = node_features[:, 0], node_features[:, 1:]
        return graph_emb, node_features

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.project:
            node_features = self.projection_in(node_features)
        if self.use_pma:
            # No CLS in sequence — transformer sees only the 36 content nodes.
            # is_encoder=False gives a plain symmetric 6×6 neighborhood mask for N=36.
            x = self.dropout(F.silu(self.fc_in(node_features)))
            x = self.graph_transformer(x, mask=None, is_encoder=False)
            node_features = self.output_norm(x)
            graph_emb = self.pma(node_features)
        else:
            # CLS mode — prepend summary node, run transformer, read out position 0.
            x, _ = self.init_message_matrix(node_features, edge_features, mask)
            x = self.graph_transformer(x, mask=None, is_encoder=True)
            x = self.output_norm(x)
            graph_emb, node_features = self.read_out_message_matrix(x)
        graph_emb = graph_emb + self.stats_correction(node_features)
        if self.structural_correction is not None:
            graph_emb = graph_emb + self.structural_correction(node_features)
        return graph_emb, node_features


class FiLMConditioner(nn.Module):
    """Projects z into per-layer (gamma, beta) pairs for FiLM conditioning.

    Uses a small shared bottleneck to keep parameter count manageable.
    Output projections are zero-initialized so FiLM is identity at the
    start of training — no change to initial optimization landscape.
    """

    def __init__(self, z_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        mid = hidden_dim // 4
        self.encode = nn.Sequential(nn.Linear(z_dim, mid), nn.SiLU())
        self.layer_projs = nn.ModuleList([
            nn.Linear(mid, 2 * hidden_dim) for _ in range(num_layers)
        ])
        for proj in self.layer_projs:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, z: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        h = self.encode(z)  # [B, mid]
        params = []
        for proj in self.layer_projs:
            out = proj(h)  # [B, 2*hidden_dim]
            gamma, beta = out.chunk(2, dim=-1)  # [B, hidden_dim] each
            params.append((gamma, beta))
        return params


class GraphDecoder(torch.nn.Module):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        grid_size = getattr(hparams, "grid_size", 6)
        self.positional_embedding = PositionalEncoding(
            hparams.graph_decoder_hidden_dim, grid_size=grid_size
        )
        use_rope = getattr(hparams, "use_rope", False)
        self.graph_transformer = Transformer(
            hidden_dim=hparams.graph_decoder_hidden_dim,
            num_heads=hparams.graph_decoder_num_heads,
            ppf_hidden_dim=hparams.graph_decoder_ppf_hidden_dim,
            num_layers=hparams.graph_decoder_num_layers,
            dropout=hparams.dropout,
            output_init_std=0.1,
            # 1D RoPE over a flattened 2D grid is topologically inconsistent; position is
            # already carried by the 2D sinusoidal PE. Off by default (configurable).
            rope=LLamaRotaryEmbedding(hparams.head_dim) if use_rope else None,
            qk_norm=getattr(hparams, "qk_norm", False),
        )
        self.use_film = getattr(hparams, "use_film", False)
        if self.use_film:
            self.film = FiLMConditioner(
                z_dim=hparams.encoder_hidden_dim,  # CLS token dimension
                hidden_dim=hparams.graph_decoder_hidden_dim,
                num_layers=hparams.graph_decoder_num_layers,
            )
        mid_dim = hparams.node_z_dim * 4
        self.fc_in = nn.Sequential(
            nn.Linear(hparams.node_z_dim, mid_dim),
            nn.SiLU(),
            nn.Linear(mid_dim, hparams.graph_decoder_hidden_dim),
        )
        if hparams.project:
            self.node_fc_out = nn.Linear(
                hparams.graph_decoder_hidden_dim, hparams.num_node_features
            )
        self.project = hparams.project
        self.dropout = nn.Dropout(hparams.dropout)

    def forward(
        self,
        z_nodes: torch.Tensor,
        z_global: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = z_nodes.shape
        x = self.dropout(self.fc_in(z_nodes))  # [B, N, hidden_dim]
        x = x + self.positional_embedding(B, N)        # inject 2D grid coordinates
        film_params = self.film(z_global) if self.use_film else None
        # mask=None → neighborhood mask (6×6 grid adjacency) — local refinement with RoPE
        x = self.graph_transformer(x, mask=mask, is_encoder=False, film_params=film_params)
        if self.project:
            x = self.node_fc_out(x)
        return x, torch.empty(0, device=x.device)


# class Permuter(torch.nn.Module):
#     def __init__(self, hparams: DictConfig):
#         super().__init__()
#         self.scoring_fc = nn.Sequential(
#             nn.Linear(hparams.graph_decoder_hidden_dim, hparams.graph_decoder_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hparams.graph_decoder_hidden_dim, hparams.graph_decoder_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hparams.graph_decoder_hidden_dim, 1),
#         )
#         self.perm_context = nn.Sequential(
#             nn.Linear(hparams.grid_size**2, hparams.emb_dim),
#             nn.LayerNorm(hparams.emb_dim),
#             nn.ReLU(),
#         )
#         self.break_symmetry_scale = hparams.break_symmetry_scale

#     def score(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#         scores = self.scoring_fc(x)

#         if mask.sum() == 0:
#             fill_value = -1e6
#         else:
#             fill_value = scores.min().item() - 1

#         scores = scores.masked_fill(mask.unsqueeze(-1) == 0, fill_value)
#         return scores

#     def soft_sort(self, scores: torch.Tensor, hard: bool, tau: float) -> torch.Tensor:
#         scores_sorted = scores.sort(descending=True, dim=1)[0]
#         pairwise_diff = (scores.transpose(1, 2) - scores_sorted).abs().neg() / tau
#         perm = pairwise_diff.softmax(-1)
#         if hard:
#             perm_ = torch.zeros_like(perm, device=perm.device)
#             perm_.scatter_(-1, perm.topk(1, -1)[1], value=1)
#             perm = (perm_ - perm).detach() + perm
#         return perm

#     def mask_perm(self, perm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#         batch_size, num_nodes = mask.size(0), mask.size(1)
#         eye = torch.eye(num_nodes, num_nodes).unsqueeze(0).expand(batch_size, -1, -1).type_as(perm)

#         mask = mask.unsqueeze(-1).expand(-1, -1, num_nodes)
#         perm = torch.where(mask, perm, eye)
#         return perm

#     def forward(
#         self,
#         node_features: torch.Tensor,
#         tau: float,
#         mask: torch.Tensor,
#         hard: bool = False,
#     ) -> tuple[torch.Tensor, Any, None]:
#         # add noise to break symmetry
#         device = node_features.device
#         node_features = node_features + torch.randn_like(node_features) * self.break_symmetry_scale
#         mask = mask.to(device)
#         scores = self.score(node_features, mask)
#         context = scores.squeeze(-1)
#         context = self.perm_context(context)

#         perm = self.soft_sort(scores, hard, tau)
#         perm = perm.transpose(2, 1)
#         perm = self.mask_perm(perm, mask)
#         return perm, context, None

#     @staticmethod
#     def permute_node_features(node_features: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
#         node_features = torch.matmul(perm, node_features)
#         return node_features

#     @staticmethod
#     def permute_edge_features(edge_features: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
#         edge_features = torch.matmul(perm.unsqueeze(1), edge_features)
#         edge_features = torch.matmul(perm.unsqueeze(1), edge_features.permute(0, 2, 1, 3))
#         edge_features = edge_features.permute(0, 2, 1, 3)
#         return edge_features

#     @staticmethod
#     def permute_graph(graph: DenseGraphBatch, perm: torch.Tensor) -> DenseGraphBatch:
#         graph.node_features = Permuter.permute_node_features(graph.node_features, perm)
#         graph.edge_features = Permuter.permute_edge_features(graph.edge_features, perm)
#         return graph


# class SimplePermuter(torch.nn.Module):
#     def __init__(self, hparams: DictConfig):
#         super().__init__()
#         self.turn_off = hparams.turn_off
#         self.use_ce = hparams.use_ce
#         # self.use_context = hparams.use_context
#         self.scoring_fc = torch.nn.Linear(
#             hparams.graph_decoder_hidden_dim, hparams.num_permutations
#         )
#         # self.layer_norm = torch.nn.LayerNorm(hparams.graph_decoder_hidden_dim)
#         self.graph_transformer = Transformer(
#             hidden_dim=hparams.graph_decoder_hidden_dim,
#             num_heads=hparams.graph_decoder_num_heads,
#             ppf_hidden_dim=hparams.graph_decoder_ppf_hidden_dim,
#             num_layers=2,
#             dropout=hparams.dropout,
#             rope=LLamaRotaryEmbedding(hparams.head_dim),
#         )
#         self.perm_node = nn.Parameter(
#             torch.randn(1, 1, hparams.graph_decoder_hidden_dim)
#         )
#         # nn.init.trunc_normal_(self.perm_node, std=0.02)
#         self.spectral_embeddings = SklearnSpectralEmbedding(
#             hparams.n_components,
#             hparams.graph_decoder_hidden_dim,
#             hparams.grid_size,
#         )
#         self.perm_context = torch.nn.Linear(hparams.num_permutations, hparams.emb_dim)
#         predefined_permutations = self.create_predefine_permutations(hparams.grid_size)
#         # Predefined permutation matrices (B, num_permutations, N, N)
#         self.register_buffer("predefined_permutations", predefined_permutations)
#         self.break_symmetry_scale = hparams.break_symmetry_scale

#     def forward(
#         self,
#         node_features: torch.Tensor,
#         tau: float,
#         mask: torch.Tensor,
#         hard: bool = False,
#         labels: Optional[torch.Tensor] = None,
#     ) -> Tuple[Optional[torch.Tensor], Any, Optional[torch.Tensor]]:
#         # Add noise to break symmetry

#         device = node_features.device
#         batch_size = node_features.shape[0] // 8

#         if self.turn_off:
#             perm = self.predefined_permutations.repeat_interleave(batch_size, dim=0)
#             return perm, None, None, None
#         # TODO: Think if I need to add this break symmetry scale
#         node_features = (
#             node_features + torch.randn_like(node_features) * self.break_symmetry_scale
#         )
#         node_features = self.spectral_embeddings(node_features)
#         cls_tokens = self.perm_node.expand(batch_size * 8, -1, -1)
#         node_features = torch.cat([cls_tokens, node_features], dim=1)  # (B, N+1, D)
#         node_features = self.graph_transformer(
#             node_features, mask=None, is_encoder=True
#         )
#         # Score each permutation option
#         cls_out = node_features[:, 0, :]
#         # cls_out = self.layer_norm(cls_out)
#         scores = self.scoring_fc(cls_out)  # (B, num_permutations)
#         context = None
#         # if self.use_context:
#         #     context = self.perm_context(scores)

#         # 5) optionally supervise with cross-entropy
#         ce_loss = None
#         if self.use_ce and labels:
#             labels = labels.to(device)
#             ce_loss = F.cross_entropy(scores, labels)

#         probs, soft_probs = softmax_head(scores, tau)
#         # context = self.perm_context(probs)                            # (B*8, emb_dim)

#         perms_buffer = self.predefined_permutations.to(device)  # (P, N, N)
#         # Shape: (B, N, N) = (B, num_permutations, N, N) * (B, num_permutations, 1, 1)
#         # Expand probs to (B, num_permutations, 1, 1) to match (num_permutations, N, N)
#         probs_expanded = probs.unsqueeze(-1).unsqueeze(
#             -1
#         )  # (B, num_permutations, 1, 1)                    # (B*8, P, 1, 1)
#         perm = torch.sum(probs_expanded * perms_buffer, dim=1)  # (B, N, N)
#         return perm, context, soft_probs, ce_loss

#     def _permutation_matrix_90(self, n: int) -> torch.Tensor:
#         indices = torch.arange(n * n).reshape(n, n)
#         rotated_indices = indices.rot90(-1).reshape(-1)
#         perm = torch.eye(n * n, dtype=torch.float32)[rotated_indices]
#         return perm

#     def _y_axis_reflection_matrix(self, n: int) -> torch.Tensor:
#         indices = torch.arange(n * n).reshape(n, n)
#         reflected_indices = indices.flip(1).reshape(-1)
#         perm = torch.eye(n * n, dtype=torch.float32)[reflected_indices]
#         return perm

#     def create_predefine_permutations(self, n: int) -> torch.Tensor:
#         perm = torch.eye(n * n, dtype=torch.float32)
#         perm_90 = self._permutation_matrix_90(n)
#         perm_180 = torch.matmul(perm_90, perm_90)
#         perm_270 = torch.matmul(perm_180, perm_90)

#         perm_y_reflection = self._y_axis_reflection_matrix(n)
#         perm_y_reflection_90 = torch.matmul(perm_y_reflection, perm_90)
#         perm_y_reflection_180 = torch.matmul(perm_y_reflection, perm_180)
#         perm_y_reflection_270 = torch.matmul(perm_y_reflection, perm_270)

#         # TODO: If I shuffle the train set with every epoch can I have fix this list if I have no labels?
#         permutations = torch.stack(
#             [
#                 perm,
#                 perm_90,
#                 perm_180,
#                 perm_270,
#                 perm_y_reflection,
#                 perm_y_reflection_90,
#                 perm_y_reflection_180,
#                 perm_y_reflection_270,
#             ]
#         )
#         return permutations

#     @staticmethod
#     def permute_node_features(
#         node_features: torch.Tensor, perm: torch.Tensor
#     ) -> torch.Tensor:
#         """Apply the permutation to node features."""
#         return torch.matmul(perm, node_features)

class SimplePermuter(torch.nn.Module):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.turn_off = hparams.turn_off
        self.curriculum_epoch = getattr(hparams, "curriculum_epoch", -1)
        self.freeze_epochs = getattr(hparams, "freeze_epochs", 0)
        self.use_ce = hparams.use_ce
        self.scoring_fc = torch.nn.Linear(
            hparams.graph_decoder_hidden_dim, hparams.num_permutations
        )
        self.graph_transformer = Transformer(
            hidden_dim=hparams.graph_decoder_hidden_dim,
            num_heads=hparams.graph_decoder_num_heads,
            ppf_hidden_dim=hparams.graph_decoder_ppf_hidden_dim,
            num_layers=2,
            dropout=hparams.dropout,
            rope=LLamaRotaryEmbedding(hparams.head_dim),
        )

        self.perm_node = nn.Parameter(torch.empty(1, 1, hparams.graph_decoder_hidden_dim))
        nn.init.trunc_normal_(self.perm_node, std=0.02)

        self.spectral_embeddings = SklearnSpectralEmbedding(
            hparams.n_components,
            hparams.graph_decoder_hidden_dim,
            hparams.grid_size,
        )

        self.grid_size = hparams.grid_size
        self.num_permutations = hparams.num_permutations
        self.num_views = hparams.num_permutations  # D4 group size = num augmented views
        self.break_symmetry_scale = hparams.break_symmetry_scale

        self.register_buffer(
            "perm_matrices",
            self._precompute_permutation_matrices(hparams.grid_size)
        )

    def _precompute_permutation_matrices(self, grid_size: int) -> torch.Tensor:
        """Precompute all 8 D4 permutation matrices once at init."""
        n = grid_size
        n_nodes = n * n
        matrices = []

        # Identity
        matrices.append(torch.eye(n_nodes))

        # Rotations: 90, 180, 270 CW
        for k in range(1, 4):
            indices = torch.arange(n_nodes).reshape(n, n)
            for _ in range(k):
                indices = indices.rot90(-1)  # CW
            matrices.append(torch.eye(n_nodes)[indices.reshape(-1)])

        # Y-axis reflection
        indices = torch.arange(n_nodes).reshape(n, n)
        reflected = indices.flip(1).reshape(-1)
        reflection = torch.eye(n_nodes)[reflected]
        matrices.append(reflection)

        # Y-reflection + rotations: 90, 180, 270 CW
        for k in range(1, 4):
            indices = torch.arange(n_nodes).reshape(n, n)
            for _ in range(k):
                indices = indices.rot90(-1)  # CW
            rotation = torch.eye(n_nodes)[indices.reshape(-1)]
            matrices.append(torch.matmul(reflection, rotation))

        return torch.stack(matrices, dim=0)  # (8, n_nodes, n_nodes)

    def forward(
        self,
        node_features: torch.Tensor,
        tau: float,
        mask: torch.Tensor,
        hard: bool = False,
        labels: torch.Tensor | None = None,
    ):
        device = node_features.device
        total_batch = node_features.shape[0]
        batch_size = total_batch // 8

        if self.turn_off:
            # Return the correct permutation matrix for each augmentation.
            # Must match the data key order from IMCBaseDictTransform.keys:
            #   0: r0_f, 1: r0_nf, 2: r180_f, 3: r180_nf,
            #   4: r270_f, 5: r270_nf, 6: r90_f, 7: r90_nf
            # Convention (from apply_transform):
            #   rotation: torch.rot90(grid, k=angle//90)  [CCW]
            #   flip:     torch.flip(grid, dims=[-1])      [Y-axis, after rotation]
            n = self.grid_size
            n_nodes = n * n
            dtype = node_features.dtype

            # (num_ccw_90_rotations, do_y_flip) for each augmentation in data order
            aug_params = [
                (0, True),   # r0_f
                (0, False),  # r0_nf  (identity)
                (2, True),   # r180_f
                (2, False),  # r180_nf
                (3, True),   # r270_f
                (3, False),  # r270_nf
                (1, True),   # r90_f
                (1, False),  # r90_nf
            ]

            idx = torch.arange(n_nodes, device=device).reshape(n, n)
            fixed_perms = []
            for k, flip in aug_params:
                src = idx
                if k > 0:
                    src = torch.rot90(src, k=k, dims=[0, 1])
                if flip:
                    src = torch.flip(src, dims=[-1])
                mat = torch.eye(n_nodes, device=device, dtype=dtype)[src.reshape(-1)]
                fixed_perms.append(mat)

            fixed_perms = torch.stack(fixed_perms, dim=0)  # (8, N, N)
            perm = fixed_perms.repeat_interleave(batch_size, dim=0)  # (8*B, N, N)

            # Shadow mode: run learned forward for perm_loss pre-training.
            # The oracle perm is returned for the decoder, but soft_probs flow
            # through perm_loss so the permuter learns diversity before the switch.
            # Noise must come AFTER spectral_embeddings — content_norm inside SE
            # normalises x to unit scale, so noise added before is wiped out.
            shadow_features = self.spectral_embeddings(node_features)
            if self.break_symmetry_scale > 0:
                shadow_features = shadow_features + torch.randn_like(shadow_features) * self.break_symmetry_scale
            cls_tokens = self.perm_node.expand(total_batch, -1, -1)
            shadow_features = torch.cat([cls_tokens, shadow_features], dim=1)
            shadow_features = self.graph_transformer(shadow_features, mask=mask, is_encoder=False)
            shadow_scores = self.scoring_fc(shadow_features[:, 0, :])
            _, soft_probs = sinkhorn_head(shadow_scores, tau, num_views=self.num_views)
            return perm, None, soft_probs, None

        # Noise added AFTER spectral_embeddings so content_norm cannot cancel it.
        node_features = self.spectral_embeddings(node_features)
        if self.break_symmetry_scale > 0:
            node_features = node_features + torch.randn_like(node_features) * self.break_symmetry_scale

        cls_tokens = self.perm_node.expand(total_batch, -1, -1)
        node_features = torch.cat([cls_tokens, node_features], dim=1)

        node_features = self.graph_transformer(node_features, mask=mask, is_encoder=False) # does not matter I pass mask so 1 will be added

        cls_out = node_features[:, 0, :]
        scores = self.scoring_fc(cls_out)

        ce_loss = None
        if self.use_ce:
            # Ground truth labels are implicit in the batch layout:
            # rows 0..B-1 = class 0, B..2B-1 = class 1, ..., 7B..8B-1 = class 7.
            labels = torch.arange(self.num_permutations, device=device).repeat_interleave(batch_size)
            ce_loss = F.cross_entropy(scores, labels)

        probs, soft_probs = sinkhorn_head(scores, tau, num_views=self.num_views)
        perm = self._compute_weighted_permutation(probs)

        return perm, None, soft_probs, ce_loss

    def _compute_weighted_permutation(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute weighted sum of permutation matrices using precomputed buffers.

        Args:
            probs: (B, num_permutations)
        Returns:
            perm: (B, n_nodes, n_nodes)
        """
        # probs: (B, 8), perm_matrices: (8, N, N) -> (B, N, N)
        return torch.einsum('bp,pnm->bnm', probs, self.perm_matrices)

    @staticmethod
    def permute_node_features(node_features: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
        """Apply the permutation to node features."""
        return torch.matmul(perm, node_features)


class BottleNeckEncoder(torch.nn.Module):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.d_in = hparams.graph_encoder_hidden_dim
        self.d_out = hparams.emb_dim
        self.vae = hparams.vae
        self.num_permutations = hparams.num_permutations
        self.activation = {
            "relu": torch.nn.ReLU(),
            "gelu": torch.nn.GELU(),
            "silu": torch.nn.SiLU(),
            "gelu2": torch.nn.GELU(approximate="tanh"),
            "leaky_relu": torch.nn.LeakyReLU(negative_slope=0.01),
        }[hparams.activation.lower()]
        # Hidden layer THEN projection to mu/logvar
        self.fc_hidden = nn.Linear(self.d_in, self.d_in)
        if self.vae:
            self.w = nn.Linear(self.d_in, 2 * self.d_out)
        else:
            self.w = nn.Linear(self.d_in, self.d_out)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # Correct order: Linear → Activation → Linear (preserves full input info)
        x = self.w(self.activation(self.fc_hidden(x)))
        if self.vae:
            batch_size = x.shape[0] // self.num_permutations
            mu = x[:, : self.d_out]
            logvar = x[:, self.d_out :]
            logvar = torch.clamp(logvar, -10, 10)  # prevents std explosion
            std = torch.exp(0.5 * logvar)
            batch_std = std[:batch_size, :]
            batch_eps = torch.randn_like(batch_std)
            eps = (
                batch_eps
                .unsqueeze(0)
                .repeat(self.num_permutations, 1, 1)
                .view(-1, batch_eps.shape[1])
            )
            x = mu + eps * std
            return x, mu, logvar
        else:
            return x, None, None


class BottleNeckDecoder(torch.nn.Module):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.d_in = hparams.emb_dim
        self.d_out = hparams.graph_decoder_hidden_dim
        self.w = nn.Linear(self.d_in, self.d_out)
        # Normalise z after projection so it arrives at the decoder with std~1,
        # preventing it from drowning out the positional embeddings when broadcast.
        self.norm = nn.LayerNorm(self.d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.w(x))


# def softmax_head(scores: torch.Tensor, tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#     # Softmax over scores to get probabilities for each permutation
#     soft_probs = torch.softmax(scores / tau, dim=-1)  # (B, num_permutations)
#     # Hard selection using Gumbel-Softmax (discrete but differentiable)
#     one_hot = torch.zeros_like(soft_probs)
#     one_hot.scatter_(1, soft_probs.argmax(dim=-1, keepdim=True), 1.0)
#     probs = (one_hot - soft_probs).detach() + soft_probs

#     return probs, soft_probs


def softmax_head(
    scores: torch.Tensor, tau: float, training: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    if training:
        # Sample Gumbel noise: -log(-log(U))
        unif = torch.clamp(torch.rand_like(scores), min=1e-20, max=1 - 1e-20)
        gumbel_noise = -torch.log(-torch.log(unif))
        logits = (scores + gumbel_noise) / tau
    else:
        logits = scores / tau

    soft_probs = torch.softmax(logits, dim=-1)

    # Straight-Through Logic
    one_hot = torch.zeros_like(soft_probs).scatter_(1, soft_probs.argmax(dim=-1, keepdim=True), 1.0)
    probs = (one_hot - soft_probs).detach() + soft_probs

    return probs, soft_probs


def sinkhorn_normalization(
    log_alpha: torch.Tensor, n_iters: int = 20
) -> torch.Tensor:
    """Sinkhorn-Knopp iterations on a log-domain [B, n_views, n_classes] tensor.

    Produces a doubly-stochastic matrix: each view sums to 1 across classes
    (row norm) and each class sums to 1 across views (column norm). This
    jointly enforces a bijective assignment — no two views of the same image
    collapse onto the same class — without any explicit diversity loss.
    """
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)  # rows
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)  # cols
    return torch.exp(log_alpha)


def sinkhorn_head(
    scores: torch.Tensor,
    tau: float,
    num_views: int,
    n_iters: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sinkhorn-based permutation head.

    Jointly assigns all `num_views` views of each image to distinct classes
    via doubly-stochastic normalisation.  Straight-through estimator makes
    the hard argmax selection differentiable.

    Args:
        scores:    [num_views * B, num_classes]  — raw logits from scoring_fc
        tau:       temperature (lower = more peaked)
        num_views: number of augmented views per image (8 for D4)
        n_iters:   Sinkhorn iterations

    Returns:
        probs:      [num_views * B, num_classes]  STE-hard probs (one-hot fwd)
        soft_probs: [num_views * B, num_classes]  soft doubly-stochastic probs
    """
    total_batch, num_classes = scores.shape
    B = total_batch // num_views

    # Fall back to independent softmax when the batch isn't a clean multiple of
    # num_views (e.g. single-image inference, or odd batch sizes at epoch end).
    # Sinkhorn is a training-time diversity enforcer; the learned scores are still
    # meaningful for argmax prediction without it.
    if B == 0 or total_batch % num_views != 0:
        soft_probs = torch.softmax(scores / tau, dim=-1)
        one_hot = torch.zeros_like(soft_probs).scatter_(
            1, soft_probs.argmax(dim=-1, keepdim=True), 1.0
        )
        probs = (one_hot - soft_probs).detach() + soft_probs
        return probs, soft_probs

    # Reshape to [B, num_views, num_classes] — group views per image
    log_alpha = scores.view(num_views, B, num_classes).permute(1, 0, 2) / tau

    # Doubly-stochastic assignment via Sinkhorn
    assignment = sinkhorn_normalization(log_alpha, n_iters=n_iters)  # [B, V, C]

    # Flatten back to [num_views * B, num_classes] in original batch order
    soft_probs = assignment.permute(1, 0, 2).reshape(total_batch, num_classes)

    # Straight-through: forward = argmax one-hot, backward = soft_probs
    one_hot = torch.zeros_like(soft_probs).scatter_(
        1, soft_probs.argmax(dim=-1, keepdim=True), 1.0
    )
    probs = (one_hot - soft_probs).detach() + soft_probs

    return probs, soft_probs


def gumbel_softmax_head(
    scores: torch.Tensor, tau: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # First compute soft probabilities
    soft_probs = F.gumbel_softmax(scores, tau=tau, hard=False, dim=-1)

    # Then derive hard probabilities from soft_probs
    probs = F.one_hot(soft_probs.argmax(dim=-1), scores.size(-1)).float()

    # Apply straight-through estimator: forward uses probs, backward uses soft_probs
    probs = probs - soft_probs.detach() + soft_probs

    return probs, soft_probs
