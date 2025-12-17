import os, time, argparse, torch, yaml
import numpy as np
import pandas as pd
from training.dataset_loader import get_dataloaders  # 若不需要可删
from models.cnn_baseline import CNNBaseline

@torch.no_grad()
def bench(model, device, shape=(64,1,8,8), warmup=50, iters=200):
    x = torch.randn(*shape, device=device)
    # warmup
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    avg = dt / iters
    fps = shape[0] / avg
    return avg, fps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=str, default="1,8,32,128,512", help="逗号分隔的batch列表")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--load_ckpt", type=str, default="", help="可选：加载已训练权重的路径")
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--W", type=int, default=8)
    ap.add_argument("--C", type=int, default=1)
    ap.add_argument("--outdir", type=str, default="results/bench")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = CNNBaseline().to(device).eval()
    if args.load_ckpt:
        try:
            state = torch.load(args.load_ckpt, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(args.load_ckpt, map_location=device)
        model.load_state_dict(state)
        print(f"✅ Loaded weights: {args.load_ckpt}")

    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    rows = []
    for B in batches:
        avg, fps = bench(model, device, shape=(B, args.C, args.H, args.W),
                         warmup=args.warmup, iters=args.iters)
        rows.append({"batch": B, "latency_ms": avg*1000, "throughput_sps": fps})
        print(f"B={B:<4d} | avg latency/iter={avg*1000:.3f} ms | throughput={fps:.1f} samples/s")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.outdir, "inference_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"💾 Saved: {csv_path}")

if __name__ == "__main__":
    main()
