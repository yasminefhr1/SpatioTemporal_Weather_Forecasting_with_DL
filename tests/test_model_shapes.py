import torch
from src.models import PatchTST, Seq2SeqAttentionModel

def test_patchtst_output_shape():
    model = PatchTST(
        input_size=8,
        horizon=12,
        patch_len=4,
        d_model=64,
        n_heads=4
    )
    x = torch.randn(16, 24, 8)
    y = model(x)
    assert y.shape == (16, 12)

def test_seq2seq_output_shape():
    model = Seq2SeqAttentionModel(
        input_size=8,
        hidden_size=64,
        horizon=12
    )
    x = torch.randn(16, 24, 8)
    y = model(x)
    assert y.shape == (16, 12)