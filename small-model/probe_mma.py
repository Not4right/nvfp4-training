"""Probe: does the sm_120a NVFP4 block-scaled mma work, and what are its
fragment + scale-factor layouts?  Kernel takes raw per-lane registers so all
layout guesses live in Python."""
import numpy as np, torch, itertools
from nvrtc_util import Module, launch

SRC = r"""
extern "C" __global__ void probe(const unsigned* A, const unsigned* B, const unsigned* SA,
                                 const unsigned* SB, float* D, int bidA, int tidA, int bidB, int tidB) {
  int l = threadIdx.x;
  unsigned a0=A[l*4+0],a1=A[l*4+1],a2=A[l*4+2],a3=A[l*4+3],b0=B[l*2+0],b1=B[l*2+1];
  unsigned sa=SA[l], sb=SB[l];
  float d0,d1,d2,d3; float z=0.f;
  unsigned short ba=bidA, ta=tidA, bb=bidB, tb=tidB;
  asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {%15,%16}, %17, {%18,%19};\n"
    : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"f"(z),"f"(z),"f"(z),"f"(z),
      "r"(sa),"h"(ba),"h"(ta),"r"(sb),"h"(bb),"h"(tb));
  D[l*4+0]=d0; D[l*4+1]=d1; D[l*4+2]=d2; D[l*4+3]=d3;
}
"""
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
def ue4m3(x):  # exact for powers of two in range
    e = int(np.log2(x)) + 7
    return e << 3

mod = Module(SRC)
f = mod.fn("probe")

def pack8(codes):  # 8 nibbles, element 0 in low nibble
    v = 0
    for i, c in enumerate(codes):
        v |= int(c) << (4 * i)
    return v

def run(Acode, Bcode, SA, SB, ids=(0, 0, 0, 0)):
    """Acode: 16x64 codes, Bcode: 8x64 codes (n,k). SA/SB: 32 uint32 per lane."""
    Ar = np.zeros((32, 4), np.uint32); Br = np.zeros((32, 2), np.uint32)
    for l in range(32):
        g, t = l >> 2, l & 3
        Ar[l, 0] = pack8(Acode[g, t*8:t*8+8]); Ar[l, 1] = pack8(Acode[g+8, t*8:t*8+8])
        Ar[l, 2] = pack8(Acode[g, 32+t*8:32+t*8+8]); Ar[l, 3] = pack8(Acode[g+8, 32+t*8:32+t*8+8])
        Br[l, 0] = pack8(Bcode[g, t*8:t*8+8]); Br[l, 1] = pack8(Bcode[g, 32+t*8:32+t*8+8])
    dev = lambda x: torch.from_numpy(np.ascontiguousarray(x).view(np.int32)).cuda()
    D = torch.zeros(32 * 4, device="cuda")
    launch(f, (1,), (32,), [dev(Ar), dev(Br), dev(np.array(SA, np.uint32)), dev(np.array(SB, np.uint32)), D, *ids])
    torch.cuda.synchronize()
    D = D.cpu().numpy().reshape(32, 4)
    out = np.zeros((16, 8), np.float32)
    for l in range(32):
        g, t = l >> 2, l & 3
        out[g, 2*t], out[g, 2*t+1], out[g+8, 2*t], out[g+8, 2*t+1] = D[l]
    return out

rng = np.random.default_rng(0)
one = ue4m3(1.0); ones32 = [one * 0x01010101] * 32
Ac = rng.integers(0, 16, (16, 64)); Bc = rng.integers(0, 16, (8, 64))
ref = E2M1[Ac] @ E2M1[Bc].T
got = run(Ac, Bc, ones32, ones32)
print("fragment layout ok (unit scales):", np.allclose(ref, got), np.abs(ref - got).max())

# --- scale layout discovery: A kblock j gets distinct magnitude code -> decode which (row,kblock) a byte hits
kval = [2, 4, 6, 7]  # 1, 2, 4, 6 for kblocks 0..3
Ac = np.zeros((16, 64), int)
for j in range(4): Ac[:, 16*j:16*j+16] = kval[j]
Bc = np.full((8, 64), 2)  # B = 1.0
base = run(Ac, Bc, ones32, ones32)
for which in ("A", "B"):
    mapping = {}
    for l in range(32):
        for byte in range(4):
            s = [one * 0x01010101] * 32
            s[l] = (one * 0x01010101 & ~(0xFF << (8*byte))) | (ue4m3(2.0) << (8*byte))
            o = run(Ac, Bc, s, ones32) if which == "A" else run(Ac, Bc, ones32, s)
            diff = o - base
            hit = np.argwhere(np.abs(diff) > 1e-3)
            if len(hit):
                d = diff[hit[0][0], hit[0][1]] / 16
                kb = [1, 2, 4, 6].index(round(d))
                rows = sorted(set(hit[:, 0])) if which == "A" else sorted(set(hit[:, 1]))
                mapping[(l, byte)] = (rows, kb)
    print(which, "scale bytes that matter:")
    for k, v in sorted(mapping.items()):
        print("  lane", k[0], "byte", k[1], "-> rows" if which == "A" else "-> cols", v[0], "kblock", v[1])
