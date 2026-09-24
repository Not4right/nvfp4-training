import torch
from nvrtc_util import Module, launch
tests = [
 ("e2m1x2.f16x2", 'unsigned short r; unsigned x = 0x3E003C00u; asm("{.reg .b8 t; cvt.rn.satfinite.e2m1x2.f16x2 t, %1; cvt.u16.u8 %0, t;}" : "=h"(r) : "r"(x)); o[0]=r;'),
 ("e4m3x2.f16x2", 'unsigned short r; unsigned x = 0x3E003C00u; asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(r) : "r"(x)); o[0]=r;'),
 ("f16x2.e2m1x2", 'unsigned r; unsigned short x = 0x3D; asm("{.reg .b8 t; cvt.u8.u16 t, %1; cvt.rn.f16x2.e2m1x2 %0, t;}" : "=r"(r) : "h"(x)); o[0]=r;'),
 ("mma f16 acc", 'unsigned d0,d1; unsigned a=0x3C003C00u; asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%2,%2,%2}, {%2,%2}, {%3,%3};" : "=r"(d0),"=r"(d1) : "r"(a), "r"(0u)); o[0]=d0;'),
 ("ex2.approx.f16x2", 'unsigned r; asm("ex2.approx.f16x2 %0, %1;" : "=r"(r) : "r"(0x3C003C00u)); o[0]=r;'),
 ("tanh.approx.f16x2", 'unsigned r; asm("tanh.approx.f16x2 %0, %1;" : "=r"(r) : "r"(0x3C003C00u)); o[0]=r;'),
]
for name, body in tests:
    src = 'extern "C" __global__ void k(unsigned* o){' + body + '}'
    try:
        m = Module(src); o = torch.zeros(1, dtype=torch.int32, device="cuda")
        launch(m.fn("k"), (1,), (32,), [o]); torch.cuda.synchronize()
        print(f"{name:22s} OK  {o.item() & 0xFFFFFFFF:#010x}")
    except Exception as e:
        print(f"{name:22s} FAIL {[l for l in str(e).splitlines() if 'error' in l][:1]}")
