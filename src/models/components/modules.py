from typing import Any

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


class GraphAE(torch.nn.Module):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.input_size = hparams.input_size
        self.vae = hparams.vae
        self.encoder = GraphEncoder(hparams.encoder)
        self.bottle_neck_encoder = BottleNeckEncoder(hparams.bottle_neck_encoder)
        self.bottle_neck_decoder = BottleNeckDecoder(hparams.bottle_neck_decoder)
        self.permuter = SimplePermuter(hparams.permuter)
        self.decoder = GraphDecoder(hparams.decoder)

    def encode(
        self, graph: DenseGraphBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        node_features = graph.node_features
        edge_features = graph.edge_features
        mask = graph.mask
        graph_emb, node_features = self.encoder(
            node_features=node_features,
            edge_features=edge_features,
            mask=mask,
        )
        graph_emb, mu, logvar = self.bottle_neck_encoder(graph_emb)
        return graph_emb, node_features, mu, logvar

    def decode(
        self,
        graph_emb: torch.Tensor,
        perm: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> DenseGraphBatch:
        z = graph_emb  # [B, emb_dim] — raw latent for FiLM conditioning
        graph_emb = self.bottle_neck_decoder(graph_emb)
        node_logits, edge_logits = self.decoder(graph_emb=graph_emb, perm=perm, mask=mask, z=z)
        graph_pred = DenseGraphBatch(
            node_features=node_logits,
            edge_features=edge_logits,
            mask=mask,
            properties=torch.tensor([]),
        )
        return graph_pred

    def forward(self, graph: DenseGraphBatch, training: bool, tau: float = 1.0) -> tuple:
        graph_emb, node_features, mu, logvar = self.encode(graph=graph)
        perm, context, soft_probs, _ = self.permuter(
            node_features, mask=graph.mask, hard=not training, tau=tau
        )
        if context is not None:
            graph_emb += context
        graph_pred = self.decode(graph_emb, perm, graph.mask)
        return graph_emb, graph_pred, soft_probs, perm, mu, logvar


class GraphEncoder(torch.nn.Module):
    def __init__(self, hparams: DictConfig):
        super().__init__()

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
        )
        self.fc_in = nn.Linear(hparams.graph_encoder_hidden_dim, hparams.graph_encoder_hidden_dim)
        # self.layer_norm = nn.LayerNorm(hparams.graph_encoder_hidden_dim)
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
        x, _ = self.init_message_matrix(node_features, edge_features, mask)
        x = self.graph_transformer(x, mask=None, is_encoder=True)
        graph_emb, node_features = self.read_out_message_matrix(x)
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
        grid_size = getattr(hparams, "grid_size", 0)
        self.positional_embedding = PositionalEncoding(
            hparams.graph_decoder_pos_emb_dim, grid_size=grid_size
        )
        self.graph_transformer = Transformer(
            hidden_dim=hparams.graph_decoder_hidden_dim,
            num_heads=hparams.graph_decoder_num_heads,
            ppf_hidden_dim=hparams.graph_decoder_ppf_hidden_dim,
            num_layers=hparams.graph_decoder_num_layers,
            dropout=hparams.dropout,
            output_init_std=0.02,
            rope=LLamaRotaryEmbedding(hparams.head_dim),
        )
        self.use_film = getattr(hparams, "use_film", False)
        if self.use_film:
            self.film = FiLMConditioner(
                z_dim=hparams.emb_dim,
                hidden_dim=hparams.graph_decoder_hidden_dim,
                num_layers=hparams.graph_decoder_num_layers,
            )
        self.fc_in = nn.Linear(hparams.graph_decoder_hidden_dim, hparams.graph_decoder_hidden_dim)
        if hparams.project:
            self.node_fc_out = nn.Linear(
                hparams.graph_decoder_hidden_dim, hparams.num_node_features
            )
        self.project = hparams.project
        self.dropout = nn.Dropout(hparams.dropout)
        # self.layer_norm = nn.LayerNorm(hparams.graph_decoder_hidden_dim)

        # if not self.graph_transformer.is_rope:
        #     # TODO: check what should be the dim
        #     self.embedding = torch.nn.Embedding(
        #         num_embeddings=hparams.num_embeddings,
        #         embedding_dim=hparams.graph_decoder_hidden_dim,
        #     )

    def init_message_matrix(
        self, graph_emb: torch.Tensor, perm: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        batch_size = graph_emb.size(0)

        # Get positional embeddings and permute them based on predicted permutation
        pos_emb = self.positional_embedding(batch_size, num_nodes)
        pos_emb = torch.matmul(perm, pos_emb)

        # Broadcast graph_emb to every node position and add positional structure.
        # Each node token carries both content (from graph_emb) and spatial location
        # (from pos_emb), giving the decoder a much shorter credit-assignment path
        # than routing content from a single CLS token through attention.
        graph_emb_broadcast = graph_emb.unsqueeze(1).expand(-1, num_nodes, -1)
        node_tokens = graph_emb_broadcast + pos_emb  # (B, num_nodes, D)

        # Prepend the z token as a communication hub; read_out_message_matrix skips it.
        x = torch.cat([graph_emb.unsqueeze(1), node_tokens], dim=1)  # (B, 1+num_nodes, D)
        x = F.silu(self.fc_in(x))
        x = self.dropout(x)

        return x

    def read_out_message_matrix(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        node_features = x[:, 1:]  # skip z token
        if self.project:
            node_features = self.node_fc_out(node_features)
        edge_features = torch.empty(0, device=x.device)
        return node_features, edge_features

    def forward(
        self,
        graph_emb: torch.Tensor,
        perm: torch.Tensor,
        mask: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.init_message_matrix(graph_emb, perm, num_nodes=mask.size(1))
        film_params = self.film(z if z is not None else graph_emb) if self.use_film else None
        x = self.graph_transformer(x, mask=mask, is_encoder=False, film_params=film_params)
        node_features, edge_features = self.read_out_message_matrix(x)
        return node_features, edge_features


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
            shadow_features = node_features + torch.randn_like(node_features) * min(self.break_symmetry_scale, 0.1)
            shadow_features = self.spectral_embeddings(shadow_features)
            cls_tokens = self.perm_node.expand(total_batch, -1, -1)
            shadow_features = torch.cat([cls_tokens, shadow_features], dim=1)
            shadow_features = self.graph_transformer(shadow_features, mask=mask, is_encoder=False)
            shadow_scores = self.scoring_fc(shadow_features[:, 0, :])
            _, soft_probs = sinkhorn_head(shadow_scores, tau, num_views=self.num_views)
            return perm, None, soft_probs, None

        # Add noise to break symmetry
        if self.break_symmetry_scale > 0.1:
            import warnings
            warnings.warn(
                f"break_symmetry_scale={self.break_symmetry_scale} capped to 0.1"
            )
        noise_scale = min(self.break_symmetry_scale, 0.1)
        node_features = node_features + torch.randn_like(node_features) * noise_scale

        node_features = self.spectral_embeddings(node_features)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x)


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
