"""INT8 de-risk microbench (DONE; result: NOT worth it in PyTorch on Pi5 A76).

Dynamic + static qnnpack INT8 Linear vs fp32 over representative denoiser GEMM
shapes. Result: 0.68x-1.13x (best 1.13x, some SLOWER). fp32 baseline (oneDNN/ACL)
already well-optimized; qnnpack INT8 doesn't beat it consistently despite dotprod.
Conclusion: PyTorch CPU INT8 is a dead end here. A real INT8 win would need a
different runtime (ONNX Runtime / TFLite / ggml) via model export.

    OMP_NUM_THREADS=4 .venv/bin/python parallel/int8_microbench.py
"""
import time, warnings
warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from torch.ao.quantization import (QuantStub, DeQuantStub, get_default_qconfig,
                                   prepare, convert)
torch.set_num_threads(4)
torch.backends.quantized.engine = "qnnpack"


def bench(layer, x, n=15):
    with torch.no_grad():
        for _ in range(3):
            layer(x)
        t = time.perf_counter()
        for _ in range(n):
            layer(x)
    return (time.perf_counter() - t) / n * 1000


class _Static(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.q, self.lin, self.dq = QuantStub(), nn.Linear(din, dout), DeQuantStub()

    def forward(self, x):
        return self.dq(self.lin(self.q(x)))


if __name__ == "__main__":
    print(f"{'shape':22} {'fp32':>8} {'dyn-int8':>9} {'stat-int8':>10}")
    for din, dout, S in [(896, 896, 4096), (256, 1536, 4096), (512, 512, 4096)]:
        fp = nn.Linear(din, dout).eval()
        x = torch.randn(2, S, din)
        dyn = torch.quantization.quantize_dynamic(fp, {nn.Linear}, dtype=torch.qint8)
        st = _Static(din, dout).eval()
        st.lin.load_state_dict(fp.state_dict())
        st.qconfig = get_default_qconfig("qnnpack")
        prepare(st, inplace=True); st(x); convert(st, inplace=True)
        f, d, s = bench(fp, x), bench(dyn, x), bench(st, x)
        print(f"{str((din,dout,S)):22} {f:8.1f} {f/d:8.2f}x {f/s:9.2f}x")
