import time
import torch
from src.models import PatchTST

def main():
    model = PatchTST(
        input_size=8,
        horizon=12,
        patch_len=4,
        d_model=64,
        n_heads=4
    )
    model.eval()

    x = torch.randn(1, 24, 8)

    for _ in range(20):
        _ = model(x)

    start = time.perf_counter()
    for _ in range(200):
        _ = model(x)
    end = time.perf_counter()

    avg_ms = (end - start) / 200 * 1000
    print(f"Average inference latency: {avg_ms:.3f} ms/sample")

if __name__ == "__main__":
    main()