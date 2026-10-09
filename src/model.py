"""
src/model.py -- Physics-Informed Dual-Branch GAT architecture (Phase 8).

Diabetic foot thermography project. Written out programmatically by
07_model_architecture.ipynb Section 9 (via inspect.getsource on the live classes/function
defined in that notebook's Sections 2-6), so this file and the notebook stay in sync.
Phases 9-12 should import from here rather than redefining anything.

Design notes:
- GroupNorm is used everywhere a normalization layer is needed (never BatchNorm1d/2d),
  because Phase 9's MC-Dropout needs dropout layers to behave correctly under repeated
  stochastic passes, sometimes on small or single-sample batches -- BatchNorm's running/
  batch statistics are unreliable in that regime and would silently corrupt uncertainty
  estimates.
- nn.Dropout / nn.Dropout2d are used as explicit modules (never F.dropout called without a
  module), since MC-Dropout needs to find and toggle these layers by type in Phase 9.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sparse_to_dense_graph(edge_index, edge_attr, num_nodes=4):
    """Converts a batch of identical-topology sparse graphs into dense adjacency + edge
    attribute tensors, with self-loops added.

    Args:
        edge_index: (B, 2, E) int64 tensor, identical across the batch dimension.
        edge_attr:  (B, E, edge_dim) float32 tensor -- per-sample edge feature VALUES
            (these vary per subject+foot even though the topology doesn't).
        num_nodes: number of nodes per graph (always 4 for this project: LCA, LPA, MPA, MCA).

    Returns:
        dense_mask: (B, num_nodes, num_nodes) bool tensor. True at (i, j) if there is an
            edge from j to i in the original topology, OR i == j (self-loop).
        dense_edge_attr: (B, num_nodes, num_nodes, edge_dim) float32 tensor.
    """
    B, two, E = edge_index.shape
    assert two == 2, f"sparse_to_dense_graph: edge_index second dim must be 2, got {two}"
    edge_dim = edge_attr.shape[-1]

    # Topology is identical across every sample in the batch -- confirm this assumption
    # before collapsing the batch dimension away and using only the first sample's edge_index.
    same_topology = bool((edge_index == edge_index[0:1]).all())
    assert same_topology, (
        "sparse_to_dense_graph: edge_index differs across the batch -- topology is expected "
        "to be identical for every sample (only node/edge feature VALUES should vary)."
    )

    ei = edge_index[0]        # (2, E), shared topology
    src, dst = ei[0], ei[1]   # (E,), (E,)

    device = edge_index.device
    dense_mask = torch.zeros(B, num_nodes, num_nodes, dtype=torch.bool, device=device)
    dense_edge_attr = torch.zeros(B, num_nodes, num_nodes, edge_dim, dtype=edge_attr.dtype, device=device)

    # Scatter each edge's attribute into its (i, j) position. src/dst are adjacent advanced
    # indices flanked only by slices, so the result keeps the batch dim in place: (B, E, edge_dim),
    # matching edge_attr exactly.
    dense_mask[:, src, dst] = True
    dense_edge_attr[:, src, dst, :] = edge_attr

    # Self-loops: every node attends to itself too. Required here since edge_index_base/diag
    # (Phase 6) contain no self-edges.
    node_idx = torch.arange(num_nodes, device=device)
    dense_mask[:, node_idx, node_idx] = True
    dense_edge_attr[:, node_idx, node_idx, :] = 0.0  # self-loop edge_attr = [0.0, 0.0]

    return dense_mask, dense_edge_attr


class EdgeGATLayer(nn.Module):
    """One edge-conditioned graph attention layer. See markdown above for the formulation."""

    def __init__(self, in_dim, out_dim, edge_dim=2, heads=4, dropout=0.1, concat=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.heads = heads
        self.concat = concat

        self.W_node = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.W_edge = nn.Linear(edge_dim, heads * out_dim, bias=False)

        # Learned attention vector 'a' per head, applied as a dot product with the
        # concatenated [h_i ; h_j ; edge_ij] feature to produce a single scalar score.
        self.attn = nn.Parameter(torch.empty(heads, 3 * out_dim))
        nn.init.xavier_uniform_(self.attn)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features, dense_mask, dense_edge_attr):
        """
        node_features:   (B, N, in_dim)
        dense_mask:      (B, N, N) bool
        dense_edge_attr: (B, N, N, edge_dim)
        Returns: (B, N, out_dim * heads) if self.concat else (B, N, out_dim)
        """
        B, N, _ = node_features.shape
        H, D = self.heads, self.out_dim

        h = self.W_node(node_features).view(B, N, H, D)              # (B, N, H, D)
        e = self.W_edge(dense_edge_attr).view(B, N, N, H, D)         # (B, N, N, H, D)

        h_i = h.unsqueeze(2).expand(B, N, N, H, D)   # h_i broadcast over neighbor dim j
        h_j = h.unsqueeze(1).expand(B, N, N, H, D)   # h_j broadcast over node dim i

        concat_feat = torch.cat([h_i, h_j, e], dim=-1)                  # (B, N, N, H, 3D)
        score = torch.einsum("bijhd,hd->bijh", concat_feat, self.attn)  # (B, N, N, H)
        score = self.leaky_relu(score)

        mask_exp = dense_mask.unsqueeze(-1).expand(B, N, N, H)        # (B, N, N, H)
        score = score.masked_fill(~mask_exp, float("-inf"))

        attn_weights = F.softmax(score, dim=2)   # softmax over neighbor dim j, for each node i
        attn_weights = self.dropout(attn_weights)

        out = torch.einsum("bijh,bjhd->bihd", attn_weights, h)        # (B, N, H, D)

        if self.concat:
            out = out.reshape(B, N, H * D)
        else:
            out = out.mean(dim=2)

        return out


class GraphBranch(nn.Module):
    """2-layer edge-conditioned GAT (multi-head then single-head), GroupNorm + Dropout
    between them, mean-pool readout over the node dimension.
    """

    def __init__(self, node_in_dim=6, hidden_dim=32, dropout_p=0.1):
        super().__init__()
        self.gat1 = EdgeGATLayer(in_dim=node_in_dim, out_dim=hidden_dim, edge_dim=2,
                                  heads=4, dropout=dropout_p, concat=True)
        self.norm = nn.GroupNorm(num_groups=4, num_channels=hidden_dim * 4)
        self.dropout = nn.Dropout(dropout_p)
        self.gat2 = EdgeGATLayer(in_dim=hidden_dim * 4, out_dim=hidden_dim, edge_dim=2,
                                  heads=1, dropout=dropout_p, concat=False)
        self.out_dim = hidden_dim

    def forward(self, node_features, dense_mask, dense_edge_attr):
        x = self.gat1(node_features, dense_mask, dense_edge_attr)   # (B, N, hidden_dim*4)
        x = x.permute(0, 2, 1)                                      # (B, hidden_dim*4, N) for GroupNorm
        x = self.norm(x)
        x = x.permute(0, 2, 1)                                      # back to (B, N, hidden_dim*4)
        x = self.dropout(x)
        x = self.gat2(x, dense_mask, dense_edge_attr)               # (B, N, hidden_dim)
        x = x.mean(dim=1)                                           # mean-pool over nodes -> (B, hidden_dim)
        return x


class ImageBranch(nn.Module):
    """Small CNN: 3x [Conv2d(stride=2) -> GroupNorm -> ReLU -> Dropout2d], channels
    in_channels -> 16 -> 32 -> 64, global average pool to (B, 64).
    """

    def __init__(self, in_channels, dropout_p=0.1):
        super().__init__()
        channels = [in_channels, 16, 32, 64]
        blocks = []
        for i in range(3):
            blocks.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=4, num_channels=channels[i + 1]),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_p),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = channels[-1]

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)      # (B, 64, 1, 1)
        x = x.flatten(1)      # (B, 64)
        return x


class DualBranchGAT(nn.Module):
    """Fusion model: graph branch and/or image branch, concatenated, fed through an MLP
    classifier head that outputs a single logit.
    """

    def __init__(self, use_graph_branch: bool = True, use_image_branch: bool = True,
                 image_in_channels: int = 3, graph_hidden_dim: int = 32,
                 image_embed_dim: int = 64, dropout_p: float = 0.3,
                 node_in_dim: int = 6, edge_dim: int = 2):
        super().__init__()
        if not use_graph_branch and not use_image_branch:
            raise ValueError(
                "DualBranchGAT: at least one of use_graph_branch or use_image_branch must be True."
            )
        self.use_graph_branch = use_graph_branch
        self.use_image_branch = use_image_branch

        combined_dim = 0
        if use_graph_branch:
            self.graph_branch = GraphBranch(node_in_dim=node_in_dim, hidden_dim=graph_hidden_dim,
                                             dropout_p=dropout_p)
            combined_dim += graph_hidden_dim
        if use_image_branch:
            self.image_branch = ImageBranch(in_channels=image_in_channels, dropout_p=dropout_p)
            combined_dim += image_embed_dim

        hidden = max(combined_dim // 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch, graph_topology="base", image_key="colormap_image"):
        embeddings = []

        if self.use_graph_branch:
            if graph_topology == "base":
                edge_index = batch["edge_index_base"]
                edge_attr = batch["edge_attr_base"]
            elif graph_topology == "diag":
                edge_index = batch["edge_index_diag"]
                edge_attr = batch["edge_attr_diag"]
            else:
                raise ValueError(
                    f"DualBranchGAT: unknown graph_topology '{graph_topology}', expected 'base' or 'diag'."
                )
            num_nodes = batch["node_features"].shape[1]
            dense_mask, dense_edge_attr = sparse_to_dense_graph(edge_index, edge_attr, num_nodes=num_nodes)
            graph_embed = self.graph_branch(batch["node_features"], dense_mask, dense_edge_attr)
            embeddings.append(graph_embed)

        if self.use_image_branch:
            image = batch[image_key]
            image_embed = self.image_branch(image)
            embeddings.append(image_embed)

        combined = torch.cat(embeddings, dim=-1)
        logit = self.classifier(combined).squeeze(-1)   # (B,)
        return logit
