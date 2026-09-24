import torch, time, sys
import torch.nn.functional as TF
import fusedops as F

if __name__ == '__main__':
    torch.manual_seed(0)
    dev = "cuda"
    M = 16384


    def cos(a, b):
        a, b = a.float().flatten(), b.float().flatten()
        return (a @ b / (a.norm() * b.norm())).item()


    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm()).item()


    def bench(f, n=50):
        for _ in range(3): f()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n): f()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e6


    w1 = torch.randn(512, 128, device=dev) / 128 ** .5
    w2 = torch.randn(128, 512, device=dev) / 512 ** .5
    W1 = F.PackedW(w1, ffN=True); W2 = F.PackedW(w2, ffK=True)
    print("weight quant rel err", rel(W1.dequant(), w1), rel(W2.dequant(), w2))

    x1 = torch.randn(M, 128, device=dev).bfloat16()
    lnw = (1 + 0.1 * torch.randn(128, device=dev))
    y = F.mlp_fwd(x1, lnw, W1, W2)
    h = TF.layer_norm(x1.float(), (128,), lnw)
    ref_branch = TF.gelu(h @ W1.dequant().t(), approximate="tanh") @ W2.dequant().t()
    br = y.float() - x1.float()
    print(f"mlp_fwd branch: cos {cos(br, ref_branch):.5f}  rel {rel(br, ref_branch):.4f}")
    ex_branch = TF.gelu(h @ w1.t(), approximate="tanh") @ w2.t()
    print(f"   vs full-precision weights: rel {rel(br, ex_branch):.4f}")

    us = bench(lambda: F.mlp_fwd(x1, lnw, W1, W2))
    print(f"mlp_fwd: {us:.1f} us  ({M*(2*128+2*128)/us/1e3:.0f} GB/s min traffic, {2*M*2*128*512/us/1e6:.1f} TFLOPS)")
    xb = x1.clone(); w1b, w2b, lb = w1.bfloat16(), w2.bfloat16(), lnw.bfloat16()
    def torch_mlp():
        return xb + TF.gelu(TF.layer_norm(xb, (128,), lb) @ w1b.t(), approximate="tanh") @ w2b.t()
    tc = torch.compile(torch_mlp)
    if __name__ == "__main__": print(f"torch bf16 eager: {bench(torch_mlp):.1f} us   compiled: {bench(tc):.1f} us")

    import qref
    hq = qref.q_rows(h)
    u = hq @ W1.dequant().t()
    a = TF.gelu(u, approximate="tanh")
    hn = qref.torch.linalg.vector_norm(h, dim=1, keepdim=True)
    bound = (hn * W1.rn).clamp(min=0.2)
    aq = qref.q_rows_gs(a, bound / 2688.0, perm=True)
    exact_branch = aq @ W2.dequant().t()
    print(f"mlp_fwd vs exact-quant emulation: rel {rel(br, exact_branch):.5f} (bf16 output rounding ~0.003)")
