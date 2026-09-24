"""Launchers for attn8.cu (FP8 flash attention)."""
import os, torch
from nvrtc_util import Module, launch
_mod = None
def fn(name, smem=0):
    global _mod
    if _mod is None:
        _mod = Module(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "attn8.cu")).read())
    return _mod.fn(name, smem)
FWD_SMEM = 2 * (64 * 128 + 128 * 64)

def prep(qkv, qn, kn, cos, sin, B, T, H, KV, eps=1e-5):
    dev = qkv.device
    Q8 = torch.empty(B, H, T, 128, dtype=torch.uint8, device=dev)
    K8 = torch.empty(B, KV, T, 128, dtype=torch.uint8, device=dev)
    VT8 = torch.empty(B, KV, 128, T, dtype=torch.uint8, device=dev)
    vam = torch.zeros(B, KV, 128, dtype=torch.int32, device=dev)
    sv = torch.empty(B, KV, 128, dtype=torch.float32, device=dev)
    nw = B * T * (H + 2 * KV)
    launch(fn("attn_prep"), ((nw + 7) // 8,), (256,), [qkv, qn, kn, cos, sin, Q8, K8, vam, B, T, H, KV, float(eps)])
    launch(fn("v_prep"), (T // 64, B * KV), (256,), [qkv, vam, VT8, sv, B, T, H, KV])
    return Q8, K8, VT8, sv

def fwd(Q8, K8, VT8, sv, B, T, H, KV):
    O = torch.empty(B, T, H, 128, dtype=torch.bfloat16, device=Q8.device)
    L = torch.empty(B, H, T, dtype=torch.float32, device=Q8.device)
    launch(fn("attn_fwd8", FWD_SMEM), (T // 64, B * H), (128,), [Q8, K8, VT8, sv, O, L, T, H, KV, 128 ** -0.5], smem=FWD_SMEM)
    return O, L

def bwd(dO, O, LSE2, Q8, K8, qkv_t, dsamax_use, dsamax_cur, B, T, H, KV):
    """dO, O: bf16 [B,T,H,128]; qkv_t: true bf16 [B*T,(H+2KV)*128] (V is read from it).
    Returns dq fp32 [B,H,T,128], dk bf16 [B,KV,T,128], dv bf16 [B,KV,T,128] (w.r.t. normed+roped q,k and v)."""
    dev = dO.device
    D = torch.empty(B, H, T, dtype=torch.float32, device=dev)
    doam = torch.zeros(B, KV, 128, dtype=torch.int32, device=dev)
    sdo = torch.empty(B, KV, 128, dtype=torch.float32, device=dev)
    nw = B * H * (T // 64)
    launch(fn("bwd_prep1"), ((nw + 7) // 8,), (256,), [dO, O, D, doam, B, T, H, KV])
    dOT8 = torch.empty(B, H, 128, T, dtype=torch.uint8, device=dev)
    QT8 = torch.empty(B, H, 128, T, dtype=torch.uint8, device=dev)
    KT8 = torch.empty(B, KV, 128, T, dtype=torch.uint8, device=dev)
    launch(fn("bwd_transpose"), (T // 64, B * H), (256,), [dO, dOT8, doam, sdo, B, T, H, 0, H // KV])
    launch(fn("bwd_transpose"), (T // 64, B * H), (256,), [Q8, QT8, 0, 0, B, T, H, 1, 1])
    launch(fn("bwd_transpose"), (T // 64, B * KV), (256,), [K8, KT8, 0, 0, B, T, KV, 2, 1])
    dq = torch.zeros(B, H, T, 128, dtype=torch.float32, device=dev)
    dk = torch.empty(B, KV, T, 128, dtype=torch.bfloat16, device=dev)
    dv = torch.empty(B, KV, T, 128, dtype=torch.bfloat16, device=dev)
    S = (H + 2 * KV) * 128
    vptr = qkv_t[:, (H + KV) * 128:]
    launch(fn("attn_bwd8", BWD_SMEM), (T // 64, B * KV), (128,),
           [Q8, QT8, K8, KT8, vptr, S, dO, dOT8, sdo, LSE2, D, dq, dk, dv, dsamax_use, dsamax_cur, T, H, KV, 128 ** -0.5], smem=BWD_SMEM)
    return dq, dk, dv
BWD_SMEM = 8192 + 8192 + 16384 + 2048 + 2 * (4096 + 4096 + 8192 + 4096 + 128 + 128)
