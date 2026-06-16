from __future__ import annotations

import networkx as nx
import numpy as np
import torch
from torch import nn


def build_bipartite_edges(n_cpg: int, n_genes: int, degree: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    graph = nx.Graph()
    graph.add_nodes_from(range(n_cpg), bipartite=0)
    graph.add_nodes_from(range(n_cpg, n_cpg + n_genes), bipartite=1)
    for cpg in range(n_cpg):
        targets = rng.choice(n_genes, size=min(degree, n_genes), replace=False)
        for t in targets:
            graph.add_edge(cpg, n_cpg + int(t))
    src: list[int] = []
    dst: list[int] = []
    for u, v in graph.edges():
        src.extend([u, v])
        dst.extend([v, u])
    for node in range(n_cpg + n_genes):
        src.append(node)
        dst.append(node)
    return torch.tensor([src, dst], dtype=torch.long)


class GraphAttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, negative_slope: float = 0.2) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.transform = nn.Linear(d_model, d_model, bias=False)
        self.attn_src = nn.Parameter(torch.empty(n_heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(n_heads, self.head_dim))
        self.leaky = nn.LeakyReLU(negative_slope)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        batch, n_nodes, _ = x.shape
        src, dst = edge_index[0], edge_index[1]
        wh = self.transform(x).reshape(batch, n_nodes, self.n_heads, self.head_dim)
        wh_src = wh.index_select(1, src)
        wh_dst = wh.index_select(1, dst)
        e = self.leaky((wh_src * self.attn_src).sum(-1) + (wh_dst * self.attn_dst).sum(-1))

        idx = dst.view(1, -1, 1).expand(batch, -1, self.n_heads)
        e_max = x.new_full((batch, n_nodes, self.n_heads), float("-inf"))
        e_max = e_max.scatter_reduce(1, idx, e, reduce="amax", include_self=True)
        e_shift = (e - e_max.gather(1, idx)).exp()
        denom = x.new_zeros((batch, n_nodes, self.n_heads)).index_add(1, dst, e_shift)
        alpha = e_shift / (denom.gather(1, idx) + 1e-9)

        message = alpha.unsqueeze(-1) * wh_src
        out = x.new_zeros((batch, n_nodes, self.n_heads, self.head_dim)).index_add(1, dst, message)
        return out.reshape(batch, n_nodes, self.n_heads * self.head_dim)


class MethylationEncoder(nn.Module):
    def __init__(
        self,
        n_cpg: int,
        n_graph_genes: int,
        d_model: int,
        n_heads: int = 4,
        n_layers: int = 2,
        degree: int = 3,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.n_cpg = n_cpg
        self.n_graph_genes = n_graph_genes
        self.register_buffer(
            "edge_index", build_bipartite_edges(n_cpg, n_graph_genes, degree, seed)
        )
        self.cpg_embed = nn.Linear(1, d_model)
        self.gene_embed = nn.Parameter(torch.zeros(n_graph_genes, d_model))
        nn.init.normal_(self.gene_embed, std=0.02)
        self.layers = nn.ModuleList(
            [GraphAttentionLayer(d_model, n_heads) for _ in range(n_layers)]
        )
        self.activation = nn.ELU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, methyl: torch.Tensor) -> torch.Tensor:
        batch = methyl.shape[0]
        cpg_nodes = self.cpg_embed(methyl.unsqueeze(-1))
        gene_nodes = self.gene_embed.unsqueeze(0).expand(batch, -1, -1)
        nodes = torch.cat([cpg_nodes, gene_nodes], dim=1)
        edge_index = self.edge_index
        for layer in self.layers:
            nodes = self.activation(layer(nodes, edge_index)) + nodes
        gene_repr = nodes[:, self.n_cpg :]
        return self.norm(gene_repr.mean(dim=1))
