# Richer Invariant Latent z — Six Implementation Ideas

## Context and Problem

The current encoder (all branches) is a **pure set encoder**: a graph transformer with no positional embeddings, followed by a CLS token readout. Because no positional information enters the encoder, the CLS token captures only **composition** — which cell types and marker profiles are present, averaged over the 36 nodes. It cannot capture **spatial arrangement** — whether similar cells cluster together, what cell types are adjacent at boundaries, etc.

The bottleneck is in what information z contains, not in the decoder's ability to use it.

**Key constraint**: z must be **D4-invariant**. Any D4 rotation/reflection of the input image permutes the 36 nodes but must produce the same z, because the same tissue viewed from 8 orientations must map to the same latent point for downstream tasks (clustering, classification).

### Current architecture reference

```
src/models/components/modules.py  — GraphEncoder, BottleNeckEncoder, GraphDecoder
configs/model/model.yaml          — all hyperparameters
```

`GraphEncoder.forward()` (around line 121):
```python
x = self.graph_transformer(x, mask=None, is_encoder=True)
graph_emb, node_features = self.read_out_message_matrix(x)   # graph_emb = CLS token
node_features = self.output_norm(node_features)
return graph_emb, node_features                               # graph_emb goes to BottleNeckEncoder
```

`BottleNeckEncoder` takes `graph_emb [B*8, hidden_dim]` → `z [B*8, emb_dim]`.

All five ideas below enrich what goes INTO the bottleneck encoder, keeping the rest of the pipeline unchanged.

---

## Idea 1 — Higher-Order Pooling (Mean + Variance + Max)

### What it is

Replace the single CLS readout with three pooling operations over the 36 node features, concatenated:

```
graph_emb = concat(mean(h_i), var(h_i), max(h_i))   # [B*8, 3 × hidden_dim]
```

Then project back to `hidden_dim` before the bottleneck encoder.

### Why it helps

- **Mean** = current behaviour: captures average composition
- **Variance** = heterogeneity: high var → mixed tissue (many different cell types), low var → homogeneous tissue. This is biologically meaningful.
- **Max** = rare/extreme features: captures the most active marker or rarest cell type, which mean pooling systematically suppresses

All three are permutation-invariant (sum/max/mean over a set do not depend on ordering). D4 permutes which node is at which position but does not change the SET of node features → same mean, var, max.

### Where to implement

In `GraphEncoder.read_out_message_matrix()` and `forward()`:

```python
def read_out_message_matrix(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    node_features = x[:, 1:]  # [B, N, D] — skip CLS
    cls = x[:, 0]             # [B, D]    — existing CLS readout

    mean_pool = node_features.mean(dim=1)          # [B, D]
    var_pool  = node_features.var(dim=1)           # [B, D]
    max_pool  = node_features.max(dim=1).values    # [B, D]

    graph_emb = torch.cat([cls, mean_pool, var_pool, max_pool], dim=-1)  # [B, 4D]
    return graph_emb, node_features
```

Add a projection layer in `__init__` to map `4 × hidden_dim → hidden_dim`:

```python
self.pool_proj = nn.Linear(4 * hparams.graph_encoder_hidden_dim,
                           hparams.graph_encoder_hidden_dim)
```

And in `forward()`:
```python
graph_emb, node_features = self.read_out_message_matrix(x)
graph_emb = self.pool_proj(graph_emb)
node_features = self.output_norm(node_features)
return graph_emb, node_features
```

### Trade-offs

- **Pro**: ~20 lines of code, no new hyperparameters, guaranteed invariant
- **Con**: tripling the channel count before projection adds parameters to `pool_proj`; var is 0 for single-node inputs (not an issue here)
- **Gain**: mostly better representation of tissue heterogeneity, not arrangement

---

## Idea 2 — Multiple Readout Seeds (Set Transformer PMA)

### What it is

Replace the single `summary_node` CLS parameter with K learnable seed vectors. Each seed independently cross-attends over all 36 nodes, producing K different global summaries:

```
seeds [K, D] → cross-attention(Q=seeds, K=V=node_features) → K vectors
z_multi = concat or mean(K vectors)   # [K × D] or [D]
```

This is the **Pool by Multihead Attention (PMA)** layer from Lee et al., Set Transformer (2019).

### Why it helps

One CLS token = K=1 pool: the attention weights collapse all 36 nodes into one weighted sum. Different structural aspects compete for the same slot. With K=4 seeds, each seed can specialise:
- Seed 0: attends to dominant cell type (highest-norm nodes)
- Seed 1: attends to interface/boundary cells
- Seed 2: attends to rare cell types
- Seed 3: attends to high-expression cells

All K readouts are permutation-invariant (each is a weighted sum over the full set). Concatenating them → K-times richer z, still invariant.

### Where to implement

Remove `self.summary_node` from `GraphEncoder.__init__()`, add:

```python
self.K = 4  # or make it a hparam
self.pma_seeds = nn.Parameter(torch.empty(self.K, hparams.graph_encoder_hidden_dim))
nn.init.trunc_normal_(self.pma_seeds, std=0.02)
self.pma_attn = nn.MultiheadAttention(
    embed_dim=hparams.graph_encoder_hidden_dim,
    num_heads=hparams.graph_encoder_num_heads,
    batch_first=True,
    dropout=hparams.dropout,
)
self.pma_proj = nn.Linear(
    self.K * hparams.graph_encoder_hidden_dim,
    hparams.graph_encoder_hidden_dim
)
```

Replace `add_emb_node_and_feature` (remove CLS prepend) and update `forward()`:

```python
def forward(self, node_features, edge_features, mask):
    if self.project:
        node_features = self.projection_in(node_features)
    x = self.dropout(F.silu(self.fc_in(node_features)))    # [B, N, D], no CLS prepend
    x = self.graph_transformer(x, mask=None, is_encoder=True)
    node_features = self.output_norm(x)                    # [B, N, D]

    B = x.shape[0]
    seeds = self.pma_seeds.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]
    pooled, _ = self.pma_attn(query=seeds, key=x, value=x) # [B, K, D]
    graph_emb = self.pma_proj(pooled.flatten(1))           # [B, D]
    return graph_emb, node_features
```

Note: removing CLS prepend simplifies the encoder (no mask padding needed). The graph transformer then processes only the 36 content nodes.

### Trade-offs

- **Pro**: most principled extension of CLS pooling, directly addresses the single-summary bottleneck, proven in point cloud and graph literature
- **Con**: ~50 lines of code, adds `K × D²` parameters (for K=4, D=512 → ~1M params extra), requires removing CLS token from the transformer input
- **Gain**: captures K independent structural aspects of the tissue

---

## Idea 3 — Pairwise Cell-Type Interaction Features

### What it is

For every grid-adjacent pair (i, j), compute a pairwise interaction feature and aggregate over all edges:

```
e_ij = MLP(concat(h_i, h_j))    for (i,j) ∈ 6×6_grid_edges
z_edge = mean(e_ij) over all edges
z = concat(z_cls, z_edge)
```

### Why it helps

IMC tissue biology is fundamentally about **cell-cell interactions**: whether T-cells are adjacent to tumor cells, whether macrophages cluster at tumor boundaries, etc. These interaction patterns are exactly what makes tissue microenvironments biologically distinct.

Mean pooling completely loses this: a tissue with T-cells and tumor cells mixed together has the same mean h as one where they're spatially segregated — but the interaction features differ.

This is permutation-invariant because we aggregate (`mean`) over the full set of edges. D4 permutes which node is at which position, but it also consistently permutes the edges — the set of adjacent pairs (in terms of physical cells) is preserved by D4 transformations of the grid.

### Where to implement

In `GraphEncoder.__init__()`:

```python
# Register the fixed 6×6 grid edge list as a buffer
import networkx as nx
G = nx.grid_2d_graph(hparams.grid_size, hparams.grid_size)
nodes = list(G.nodes())
node_to_idx = {n: i for i, n in enumerate(nodes)}
edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
edge_index = torch.tensor(edges, dtype=torch.long)  # [E, 2]
self.register_buffer("edge_index", edge_index)

self.edge_mlp = nn.Sequential(
    nn.Linear(2 * hparams.graph_encoder_hidden_dim, hparams.graph_encoder_hidden_dim),
    nn.GELU(),
    nn.Linear(hparams.graph_encoder_hidden_dim, hparams.graph_encoder_hidden_dim // 2),
)
self.edge_proj = nn.Linear(
    hparams.graph_encoder_hidden_dim + hparams.graph_encoder_hidden_dim // 2,
    hparams.graph_encoder_hidden_dim,
)
```

In `forward()`, after getting `node_features` from the transformer:

```python
# node_features: [B, N, D]
h_i = node_features[:, self.edge_index[:, 0], :]  # [B, E, D]
h_j = node_features[:, self.edge_index[:, 1], :]  # [B, E, D]
edge_feats = self.edge_mlp(torch.cat([h_i, h_j], dim=-1))  # [B, E, D/2]
z_edge = edge_feats.mean(dim=1)                             # [B, D/2]

graph_emb = self.edge_proj(torch.cat([graph_emb, z_edge], dim=-1))  # [B, D]
```

### Trade-offs

- **Pro**: biologically most meaningful for IMC; captures the tissue microenvironment structure that matters most for downstream tasks; ~40 lines of code
- **Con**: doubles the number of pairwise interactions processed (60 edges for 6×6 grid); edge features are symmetric (h_i, h_j) = (h_j, h_i) only if you symmetrise — either use `sort(h_i, h_j)` or `concat(h_i+h_j, |h_i-h_j|)` for symmetry
- **Symmetrisation fix**: use `concat(h_i + h_j, (h_i - h_j).abs())` instead of `concat(h_i, h_j)` to make e_ij = e_ji

---

## Idea 4 — Weighted Graph Laplacian Spectrum

### What it is

Build a **node-similarity weighted adjacency matrix** from the encoder's node features, compute its Laplacian, and extract the top-k eigenvalues as structural descriptors:

```
W[i,j] = sim(h_i, h_j) · grid_adj[i,j]    # weighted adjacency (only grid neighbours)
L_w = D_w - W                              # weighted Laplacian
λ_1,...,λ_k = top-k eigenvalues of L_w     # permutation-invariant spectrum
z = concat(z_cls, λ_1,...,λ_k)
```

### Why it helps

The eigenvalues of a graph Laplacian encode **how smooth the node features are over the graph topology**:

- Small λ (slow modes): similar cells are clustered → homogeneous spatial structure
- Large λ (fast modes): dissimilar cells are adjacent → heterogeneous, fine-grained structure
- The spectral gap (λ_2 - λ_1) measures how strongly the tissue is spatially segregated into distinct regions

These are **exactly the structural features** that compositional pooling (mean) loses. Two tissues with identical cell-type composition but different spatial organisation will have different spectra.

The spectrum is permutation-invariant: `spec(P^T L P) = spec(L)` for any permutation matrix P.

The computation is differentiable: `torch.linalg.eigh` returns gradients through the eigenvalues.

### Where to implement

In `GraphEncoder.__init__()`:

```python
import networkx as nx
G = nx.grid_2d_graph(hparams.grid_size, hparams.grid_size)
A_grid = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32)
self.register_buffer("A_grid", A_grid)   # [N, N] fixed grid adjacency

self.n_spectral = 8  # top-k eigenvalues to keep
self.spectral_proj = nn.Linear(
    hparams.graph_encoder_hidden_dim + self.n_spectral,
    hparams.graph_encoder_hidden_dim,
)
```

In `forward()`, after getting `node_features`:

```python
# node_features: [B, N, D] — already output_norm'd
# Build soft similarity-weighted adjacency
h_norm = F.normalize(node_features, dim=-1)            # [B, N, D]
sim = torch.bmm(h_norm, h_norm.transpose(1, 2))        # [B, N, N]
W = F.relu(sim) * self.A_grid.unsqueeze(0)             # [B, N, N] — only grid neighbours
D_w = torch.diag_embed(W.sum(dim=-1))                  # [B, N, N]
L_w = D_w - W                                          # [B, N, N]

# Differentiable eigendecomposition (symmetric)
eigenvalues = torch.linalg.eigvalsh(L_w)               # [B, N] sorted ascending
lambda_k = eigenvalues[:, :self.n_spectral]            # [B, n_spectral]

graph_emb = self.spectral_proj(
    torch.cat([graph_emb, lambda_k], dim=-1)
)
```

### Trade-offs

- **Pro**: most theoretically principled; captures true spatial structure; directly differentiable; eigenvalues have known biological interpretation (spatial segregation, clustering)
- **Con**: `torch.linalg.eigvalsh` on [B, 36, 36] is fast but adds ~1ms per step; eigenvalue gradients can be numerically unstable when eigenvalues are degenerate (similar cells → near-degenerate L_w) — add a small `L_w + ε·I` regularisation
- **Regularisation fix**: `L_w = L_w + 1e-4 * torch.eye(N, device=L_w.device).unsqueeze(0)`
- **Gain**: unique structural descriptor not captured by any pooling-based approach

---

## Idea 5 — Canonical Sorted Encoding

### What it is

Sort the 36 node feature vectors by an invariant criterion (e.g., L2 norm), producing a canonical sequence, then encode that sequence with a small MLP or transformer:

```
h_sorted = sort(h_1,...,h_36, key=||h_i||₂)   # always same order for same tissue
z_sorted = MLP(flatten(h_sorted)) or Transformer(h_sorted)
z = concat(z_cls, z_sorted)
```

### Why it helps

Sorting by norm is permutation-invariant: the ranked list of cell features by magnitude is the same regardless of which D4 permutation was applied. The sorted sequence captures **rank-order structure** — which cells are the most/least expressed, how the feature magnitude falls off. This is a form of "canonical form" computation.

It's a weaker form of structural information than the spectrum (rank-by-norm is a 1D projection), but:
- It's differentiable via a straight-through estimator or soft-sort (Cuturi et al. 2019)
- It captures the distribution of features in a more complete way than moments (you see the full sorted sequence)
- Combined with Idea 1 (mean/var/max), it adds the full empirical CDF information

### Where to implement

Hard sort (non-differentiable but simple):

```python
# node_features: [B, N, D]
norms = node_features.norm(dim=-1)              # [B, N]
sort_idx = norms.argsort(dim=-1, descending=True)  # [B, N]
h_sorted = node_features[
    torch.arange(B).unsqueeze(1), sort_idx
]                                               # [B, N, D] — sorted by norm
```

Then encode:

```python
self.sort_encoder = nn.Sequential(
    nn.Linear(N * D, D),   # N=36, D=hidden_dim; or use a small transformer
    nn.GELU(),
    nn.Linear(D, D // 2),
)
self.sort_proj = nn.Linear(D + D // 2, D)
```

In `forward()`:
```python
z_sorted = self.sort_encoder(h_sorted.flatten(1))    # [B, D//2]
graph_emb = self.sort_proj(torch.cat([graph_emb, z_sorted], dim=-1))
```

For a differentiable version, use **soft-sort** (NeuralSort, Optimal Transport sort) — more complex but allows gradients to flow through the sorting operation.

### Trade-offs

- **Pro**: simple to implement with hard sort; no additional hyperparameters; interpretable (rank by expression level); differentiable versions exist
- **Con**: norm-based sorting is a weak criterion — two different cell arrangements can have identical norm rankings; hard sort is non-differentiable (stops gradients from flowing through the sort); `N × D = 36 × 512 = 18432` flattened input to `sort_encoder` is large
- **Better criterion**: sort by projection onto the Fiedler vector (second eigenvector of the fixed grid Laplacian) — this gives a spatial 1D ordering along the principal axis of the grid
- **Gain**: complementary to all other ideas; most useful when combined with Idea 1 or 2

---

## Idea 6 — Lightweight Grid-Topology Pairwise Features (No Edge Features Required)

### What it is

A lightweight variant of Idea 3 that explicitly accounts for the architectural constraint that `DenseGraphBatch.edge_features` is **zeros by design** — structural information was intentionally delegated to the spectral embeddings in the permuter.

Instead of reading edge features from the batch, precompute the 6×6 grid adjacency as a fixed `register_buffer` and index directly into the post-transformer `node_features`:

```python
h_i = node_features[:, edge_index[:, 0], :]   # [B, E, D]
h_j = node_features[:, edge_index[:, 1], :]   # [B, E, D]
edge_summary = (h_i * h_j).mean(dim=1)        # [B, D] — Hadamard, then mean over edges
graph_emb = graph_emb + proj(edge_summary)    # residual, zero-init proj
```

No `DenseGraphBatch.edge_features` is read at all. The `edge_index` buffer is a constant derived from the known 6×6 grid topology.

### Why it differs from Idea 3

| | Idea 3 | Idea 6 |
|---|---|---|
| Aggregation | `MLP(concat(h_i, h_j))` | `h_i ⊙ h_j` (Hadamard product) |
| Framing | Explicit pairwise interaction | Implicit, via contextualized h_i |
| Parameters | MLP + projection | Single projection (zero-init) |
| Edge source | Precomputed buffer (same as here) | Precomputed buffer |
| h_i content | Raw-ish node feature | Already neighborhood-contextualized |

The key insight: since `h_i` passed through a **grid-masked transformer**, it already encodes "what cell is here AND what cells are in my neighborhood". So `h_i ⊙ h_j` is richer than a raw co-occurrence count — it captures which latent dimensions co-activate in adjacent cells, where each dimension already summarises local context.

### Why it helps

The current z (after stats_correction) knows "cell type X exists, the tissue is heterogeneous" — but does not know "cell type X is adjacent to cell type Y". Two tissues with identical cell type composition but different spatial arrangement (segregated vs. mixed) will produce identical z. Edge summary captures adjacency patterns and is D4-invariant because:
- D4 permutes which physical cell occupies which grid position
- But it preserves the set of grid edges: if cell A is adjacent to cell B, they remain adjacent after any D4 rotation/reflection
- Mean pooling over all edges is invariant to the order in which we enumerate them

### Where to implement

In `GraphEncoder.__init__()`:

```python
import networkx as nx
G = nx.grid_2d_graph(grid_size, grid_size)
nodes = list(G.nodes())
node_to_idx = {n: i for i, n in enumerate(nodes)}
edges = torch.tensor([(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()], dtype=torch.long)
self.register_buffer("edge_index", edges)   # [E=60, 2] — fixed for 6×6 grid

self.edge_correction = nn.Linear(hparams.graph_encoder_hidden_dim,
                                  hparams.graph_encoder_hidden_dim, bias=False)
nn.init.zeros_(self.edge_correction.weight)  # zero-init → identity at start
```

In `GraphEncoder.forward()`, after `output_norm`:

```python
h_i = node_features[:, self.edge_index[:, 0], :]   # [B, E, D]
h_j = node_features[:, self.edge_index[:, 1], :]   # [B, E, D]
edge_summary = (h_i * h_j).mean(dim=1)             # [B, D]
graph_emb = graph_emb + self.edge_correction(edge_summary)
```

### Relation to the current residual correction chain

As of the current implementation, `GraphEncoder.forward()` already applies:
```python
graph_emb = graph_emb + self.stats_correction(node_features)   # mean + var + max
```

This idea adds a second residual on top:
```python
graph_emb = graph_emb + self.stats_correction(node_features)   # compositional stats
graph_emb = graph_emb + self.edge_correction(edge_summary)     # structural adjacency
```

Both are zero-initialized → training starts identical to the baseline, and the model learns which corrections are useful.

### Trade-offs

- **Pro**: ~15 lines of code; no extra hyperparameters; zero-initialization means safe to add; no dependency on DenseGraphBatch.edge_features; consistent with the "no explicit edges" design
- **Con**: Hadamard product is symmetric (`h_i ⊙ h_j = h_j ⊙ h_i`) so it can't capture directed interaction; weaker than Idea 3's MLP (no cross-dimension mixing between h_i and h_j)
- **When to prefer over Idea 3**: when you want to stay within the current "no edges by design" architectural constraint; when you want a minimal, zero-risk addition to test the hypothesis before committing to Idea 3
- **Gain**: encodes which cell-type features co-activate in adjacent positions — bridges the gap between "who is present" (mean pool) and "who is next to whom" (full Idea 3)

---

## Implementation Priority and Combinations

### Standalone value

| Idea | Code effort | Structural info gained | Biological relevance |
|---|---|---|---|
| 1. Mean+Var+Max | trivial | Low (no spatial) | Medium |
| 2. PMA seeds (K=4) | medium | Medium | Medium |
| 3. Pairwise interactions (MLP) | medium | High | **Very high** |
| 4. Laplacian spectrum | medium | **Very high** | High |
| 5. Canonical sort | low-medium | Low-medium | Low |
| 6. Grid-topology Hadamard (no edges) | **trivial** | Medium-high | High |

### Recommended combinations

**Minimum viable enrichment** (start here):
```
z = concat(CLS, mean, var, max) → projection   [Idea 1 only]
```

**Good single addition** (with explicit edges):
```
z = concat(CLS, z_edge_mean)                   [Idea 3, most IMC-relevant]
```

**Good single addition** (no-edge-features design constraint):
```
z = CLS + stats_correction + edge_correction   [Ideas 1+6, already partially implemented]
```

**Strong combined**:
```
z = concat(CLS_from_PMA_seeds, z_edge_mean)    [Ideas 2 + 3]
```

**Full structural z**:
```
z = concat(PMA_seeds_output, z_edge_mean, λ_k) [Ideas 2 + 3 + 4]
```

### What NOT to combine

- Ideas 3 and 4 are partially redundant: both capture cell-type adjacency patterns. Start with one, not both.
- Idea 5 adds marginal value on top of Ideas 2+3+4 — skip unless the others are already implemented and you want to experiment further.

---

## Invariance checklist for any new z component

Before adding any new component, verify:

1. **Is it a function of the SET {h_1,...,h_N}?** Not the sequence. Any sum/mean/max/sort over the full set is fine.
2. **Does it use absolute grid positions?** If yes, it breaks invariance. Use RELATIVE positions (distance between cells) or fixed topology (grid adjacency) only.
3. **Does it depend on the order in which augmentations are processed?** The shared-epsilon trick in `BottleNeckEncoder` preserves invariance across the 8 D4 views — make sure new components don't re-introduce view-dependent noise.

---

## Files to modify

| Change | File | Location |
|---|---|---|
| All ideas: add to encoder | `src/models/components/modules.py` | `GraphEncoder.__init__()`, `forward()` |
| Update hidden dim if needed | `configs/model/model.yaml` | `encoder:` block |
| Idea 2: remove CLS from transformer input | `modules.py` | `add_emb_node_and_feature()`, remove CLS prepend |
| Idea 3: grid edge buffer | `modules.py` | `GraphEncoder.__init__()` |
| Idea 4: grid adjacency buffer | `modules.py` | `GraphEncoder.__init__()` |
