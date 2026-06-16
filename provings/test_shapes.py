from __future__ import annotations

import torch

from epiresist.alembics import STATE_DIM
from epiresist.preparations import DoseResponseEncoder, MethylationEncoder, TranscriptEncoder


def test_model_output_shapes(model, batch):
    output = model(batch)
    assert output.regression.shape == (8,)
    assert output.logits.shape == (8,)
    assert output.trajectory.shape[1:] == (8, STATE_DIM)
    assert output.terminal_state.shape == (8, STATE_DIM)


def test_transcript_encoder_shape():
    encoder = TranscriptEncoder(n_genes=64, n_gene_sets=8, d_model=32, n_heads=2, n_layers=1)
    pooled, enzymes = encoder(torch.randn(4, 64))
    assert pooled.shape == (4, 32)
    assert enzymes.shape == (4, 3)
    assert torch.all(enzymes >= 0.0)


def test_methylation_encoder_shape():
    encoder = MethylationEncoder(n_cpg=48, n_graph_genes=16, d_model=32, n_heads=2, n_layers=1)
    out = encoder(torch.rand(4, 48))
    assert out.shape == (4, 32)


def test_doseresponse_encoder_shape():
    encoder = DoseResponseEncoder(n_replicates=2, d_model=32, filters=(8, 16, 32))
    out = encoder(torch.rand(4, 2, 7))
    assert out.shape == (4, 32)
