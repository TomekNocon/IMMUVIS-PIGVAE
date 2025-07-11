import torch
from typing import Tuple, Optional
from torch.nn import Linear, LayerNorm, Dropout
from torch.nn.functional import pad

# from src.models.components.custom_graph_transformer import Transformer
from src.models.components.llama_graph_transformer import Transformer
from src.models.components.emdeddings import PositionalEncoding
from src.data.components.graphs_datamodules import DenseGraphBatch
from src.models.components.spectral_embeddings import NetworkXSpectralEmbedding
from src.models.components.rotary_embedding import LLamaRotaryEmbedding

from omegaconf import DictConfig


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

    def encode(self, graph: DenseGraphBatch) -> Tuple[torch.Tensor]:
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
        mask: Optional[torch.Tensor] = None,
    ) -> DenseGraphBatch:
        graph_emb = self.bottle_neck_decoder(graph_emb)
        node_logits, edge_logits = self.decoder(
            graph_emb=graph_emb, perm=perm, mask=mask
        )
        graph_pred = DenseGraphBatch(
            node_features=node_logits,
            edge_features=edge_logits,
            mask=mask,
            properties=torch.tensor([]),
        )
        return graph_pred

    def forward(
        self, graph: DenseGraphBatch, training: bool, tau: float = 1.0
    ) -> Tuple:
        graph_emb, node_features, mu, logvar = self.encode(graph=graph)
        perm, context, soft_probs = self.permuter(
            node_features, mask=graph.mask, hard=not training, tau=tau
        )
        if context is not None:
            graph_emb += context
        graph_pred = self.decode(graph_emb, perm, graph.mask)
        return graph_emb, graph_pred, soft_probs, perm, mu, logvar


class GraphEncoder(torch.nn.Module):
    def __init__(self, hparams: DictConfig):
        super().__init__()

        self.projection_in = Linear(
            hparams.num_node_features, hparams.graph_encoder_hidden_dim
        )

        self.graph_transformer = Transformer(
            hidden_dim=hparams.graph_encoder_hidden_dim,
            num_heads=hparams.graph_encoder_num_heads,
            ppf_hidden_dim=hparams.graph_encoder_ppf_hidden_dim,
            num_layers=hparams.graph_encoder_num_layers,
            dropout=hparams.dropout,
        )
        self.fc_in = Linear(
            hparams.graph_encoder_hidden_dim, hparams.graph_encoder_hidden_dim
        )
        self.layer_norm = LayerNorm(hparams.graph_encoder_hidden_dim)
        self.dropout = Dropout(hparams.dropout)
        self.spectral_embeddings = NetworkXSpectralEmbedding(
            hparams.num_node_features, hparams.grid_size
        )

    def add_emb_node_and_feature(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        node_features = pad(node_features, (0, 0, 1, 0))
        mask = pad(mask, (1, 0), value=1)
        return node_features, edge_features, mask

    def init_message_matrix(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        node_features, edge_features, mask = self.add_emb_node_and_feature(
            node_features, edge_features, mask
        )
        x = self.layer_norm(self.dropout(self.fc_in(node_features)))
        return x, mask  # edge_mask

    def read_out_message_matrix(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        node_features = x
        graph_emb, node_features = node_features[:, 0], node_features[:, 1:]
        return graph_emb, node_features

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        mask: torch.Tensor,
        device: str = "mps",
    ) -> Tuple[torch.Tensor]:
        # node_features = self.spectral_embeddings(node_features)
        node_features = node_features.to(device)
        node_features = self.projection_in(node_features)
        x, _ = self.init_message_matrix(node_features, edge_features, mask)
        x = self.graph_transformer(x, mask=None, is_encoder=True)
        graph_emb, node_features = self.read_out_message_matrix(x)
        return graph_emb, node_features


class GraphDecoder(torch.nn.Module):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.positional_embedding = PositionalEncoding(
            hparams.graph_decoder_pos_emb_dim
        )
        # self.rope = LLamaRotaryEmbedding(dim=64)
        self.graph_transformer = Transformer(
            hidden_dim=hparams.graph_decoder_hidden_dim,
            num_heads=hparams.graph_decoder_num_heads,
            ppf_hidden_dim=hparams.graph_decoder_ppf_hidden_dim,
            num_layers=hparams.graph_decoder_num_layers,
            dropout=hparams.dropout,
            rope=LLamaRotaryEmbedding(hparams.head_dim),
        )
        self.fc_in = Linear(
            hparams.graph_decoder_hidden_dim, hparams.graph_decoder_hidden_dim
        )
        self.node_fc_out = Linear(
            hparams.graph_decoder_hidden_dim, hparams.num_node_features
        )
        self.dropout = Dropout(hparams.dropout)
        self.layer_norm = LayerNorm(hparams.graph_decoder_hidden_dim)

        if not self.graph_transformer.is_rope:
            self.embedding = torch.nn.Embedding(
                num_embeddings=200, embedding_dim=hparams.graph_decoder_hidden_dim
            )

    def init_message_matrix(
        self, graph_emb: torch.Tensor, perm: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        batch_size = graph_emb.size(0)
        x = graph_emb.unsqueeze(1).expand(-1, num_nodes, -1)
        # if not self.rope:
        pos_emb = self.positional_embedding(batch_size, num_nodes)
        if perm is not None:
            pos_emb = torch.matmul(perm, pos_emb)

        if not self.graph_transformer.is_rope:
            positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
            x = x + self.embedding(positions)

        x = x + pos_emb
        x = self.layer_norm(self.dropout(self.fc_in(x)))
        return x

    def read_out_message_matrix(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        node_features = x
        node_features = self.node_fc_out(node_features)
        edge_features = torch.tensor([])
        return node_features, edge_features

    def forward(
        self, graph_emb: torch.Tensor, perm: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        x = self.init_message_matrix(graph_emb, perm, num_nodes=mask.size(1))
        # if self.rope:
        #     self.graph_transformer.perm = perm
        x = self.graph_transformer(x, mask=None, is_encoder=False)
        node_features, edge_features = self.read_out_message_matrix(x)
        return node_features, edge_features


class Permuter(torch.nn.Module):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.scoring_fc = torch.nn.Sequential(
            Linear(hparams.graph_decoder_hidden_dim, hparams.graph_decoder_hidden_dim),
            torch.nn.ReLU(),
            Linear(hparams.graph_decoder_hidden_dim, hparams.graph_decoder_hidden_dim),
            torch.nn.ReLU(),
            Linear(hparams.graph_decoder_hidden_dim, 1),
        )
        self.perm_context = torch.nn.Sequential(
            Linear(hparams.grid_size**2, hparams.emb_dim),
            torch.nn.LayerNorm(hparams.emb_dim),
            torch.nn.ReLU(),
        )
        self.break_symmetry_scale = hparams.break_symmetry_scale

    def score(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.scoring_fc(x)

        if mask.sum() == 0:
            fill_value = -1e6
        else:
            fill_value = scores.min().item() - 1

        scores = scores.masked_fill(mask.unsqueeze(-1) == 0, fill_value)
        return scores

    def soft_sort(self, scores: torch.Tensor, hard: bool, tau: float) -> torch.Tensor:
        scores_sorted = scores.sort(descending=True, dim=1)[0]
        pairwise_diff = (scores.transpose(1, 2) - scores_sorted).abs().neg() / tau
        perm = pairwise_diff.softmax(-1)
        if hard:
            perm_ = torch.zeros_like(perm, device=perm.device)
            perm_.scatter_(-1, perm.topk(1, -1)[1], value=1)
            perm = (perm_ - perm).detach() + perm
        return perm

    def mask_perm(self, perm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes = mask.size(0), mask.size(1)
        eye = (
            torch.eye(num_nodes, num_nodes)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .type_as(perm)
        )

        mask = mask.unsqueeze(-1).expand(-1, -1, num_nodes)
        perm = torch.where(mask, perm, eye)
        return perm

    def forward(
        self,
        node_features: torch.Tensor,
        tau: float,
        mask: torch.Tensor,
        hard: bool = False,
    ) -> torch.Tensor:
        # add noise to break symmetry
        device = node_features.device
        node_features = (
            node_features + torch.randn_like(node_features) * self.break_symmetry_scale
        )
        mask = mask.to(device)
        scores = self.score(node_features, mask)
        context = scores.squeeze(-1)
        context = self.perm_context(context)

        perm = self.soft_sort(scores, hard, tau)
        perm = perm.transpose(2, 1)
        perm = self.mask_perm(perm, mask)
        return perm, context, None

    @staticmethod
    def permute_node_features(
        node_features: torch.Tensor, perm: torch.Tensor
    ) -> torch.Tensor:
        node_features = torch.matmul(perm, node_features)
        return node_features

    @staticmethod
    def permute_edge_features(
        edge_features: torch.Tensor, perm: torch.Tensor
    ) -> torch.Tensor:
        edge_features = torch.matmul(perm.unsqueeze(1), edge_features)
        edge_features = torch.matmul(
            perm.unsqueeze(1), edge_features.permute(0, 2, 1, 3)
        )
        edge_features = edge_features.permute(0, 2, 1, 3)
        return edge_features

    @staticmethod
    def permute_graph(graph: DenseGraphBatch, perm: torch.Tensor) -> DenseGraphBatch:
        graph.node_features = Permuter.permute_node_features(graph.node_features, perm)
        graph.edge_features = Permuter.permute_edge_features(graph.edge_features, perm)
        return graph


class SimplePermuter(torch.nn.Module):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.turn_off = hparams.turn_off
        self.scoring_fc = torch.nn.Linear(
            hparams.graph_decoder_hidden_dim, hparams.num_permutations
        )
        self.perm_context = torch.nn.Linear(hparams.num_permutations, hparams.emb_dim)
        predefined_permutations = self.create_predefine_permutations(hparams.grid_size)
        # Predefined permutation matrices (B, num_permutations, N, N)
        self.register_buffer("predefined_permutations", predefined_permutations)
        self.break_symmetry_scale = hparams.break_symmetry_scale

    def forward(
        self,
        node_features: torch.Tensor,
        tau: float,
        mask: torch.Tensor,
        hard: bool = False,
    ) -> torch.Tensor:
        # Add noise to break symmetry

        if self.turn_off:
            return None, None, None

        node_features = (
            node_features + torch.randn_like(node_features) * self.break_symmetry_scale
        )

        # Score each permutation option
        scores = self.scoring_fc(node_features).mean(dim=1)  # (B, num_permutations)
        context = self.perm_context(scores)

        # Softmax over scores to get probabilities for each permutation
        soft_probs = torch.softmax(scores / tau, dim=-1)  # (B, num_permutations)
        # Hard selection using Gumbel-Softmax (discrete but differentiable)
        one_hot = torch.zeros_like(soft_probs)
        one_hot.scatter_(1, soft_probs.argmax(dim=-1, keepdim=True), 1.0)
        probs = (one_hot - soft_probs).detach() + soft_probs

        # Combine predefined permutations with the probabilities
        # Shape: (B, N, N) = (B, num_permutations, N, N) * (B, num_permutations, 1, 1)
        # Expand probs to (B, num_permutations, 1, 1) to match (num_permutations, N, N)
        probs = probs.unsqueeze(-1).unsqueeze(-1)  # (B, num_permutations, 1, 1)
        # Combine predefined permutations with probabilities
        # Shape: (B, N, N) = (B, num_permutations, 1, 1) * (num_permutations, N, N) -> sum over 8
        perm = torch.sum(probs * self.predefined_permutations, dim=1)  # (B, N, N)

        return perm, context, soft_probs

    def _permutation_matrix_90(self, n: int) -> torch.Tensor:
        indices = torch.arange(n * n).reshape(n, n)
        rotated_indices = indices.rot90(-1).reshape(-1)
        perm = torch.eye(n * n, dtype=torch.float32)[rotated_indices]
        return perm

    def _y_axis_reflection_matrix(self, n: int) -> torch.Tensor:
        indices = torch.arange(n * n).reshape(n, n)
        reflected_indices = indices.flip(1).reshape(-1)
        perm = torch.eye(n * n, dtype=torch.float32)[reflected_indices]
        return perm

    def create_predefine_permutations(self, n: int) -> torch.Tensor:
        perm = torch.eye(n * n, dtype=torch.float32)
        perm_90 = self._permutation_matrix_90(n)
        perm_180 = torch.matmul(perm_90, perm_90)
        perm_270 = torch.matmul(perm_180, perm_90)

        perm_y_reflection = self._y_axis_reflection_matrix(n)
        perm_y_reflection_90 = torch.matmul(perm_y_reflection, perm_90)
        perm_y_reflection_180 = torch.matmul(perm_y_reflection, perm_180)
        perm_y_reflection_270 = torch.matmul(perm_y_reflection, perm_270)

        permutations = torch.stack(
            [
                perm,
                perm_90,
                perm_180,
                perm_270,
                perm_y_reflection,
                perm_y_reflection_90,
                perm_y_reflection_180,
                perm_y_reflection_270,
            ]
        )
        return permutations

    @staticmethod
    def permute_node_features(
        node_features: torch.Tensor, perm: torch.Tensor
    ) -> torch.Tensor:
        """Apply the permutation to node features."""
        return torch.matmul(perm, node_features)


class BottleNeckEncoder(torch.nn.Module):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.d_in = hparams.graph_encoder_hidden_dim
        self.d_out = hparams.emb_dim
        self.vae = hparams.vae
        self.activation = {
            "relu": torch.nn.ReLU(),
            "gelu": torch.nn.GELU(),
            "silu": torch.nn.SiLU(),
        }[hparams.activation.lower()]
        if self.vae:
            self.w = Linear(self.d_in, 2 * self.d_out)
        else:
            self.w = Linear(self.d_in, self.d_out)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        x = self.w(self.activation(x))
        if self.vae:
            mu = x[:, : self.d_out]
            logvar = x[:, self.d_out :]
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            x = mu + eps * std
            return x, mu, logvar
        else:
            return x, None, None


class BottleNeckDecoder(torch.nn.Module):
    def __init__(self, hparams: DictConfig):
        self.d_in = hparams.emb_dim
        self.d_out = hparams.graph_decoder_hidden_dim
        super().__init__()
        self.w = Linear(self.d_in, self.d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w(x)
        return x
