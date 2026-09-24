import torch
from nvrtc_util import Module, launch
for name, body in [
 ("rn_e2m1x2", 'unsigned short r; asm("{.reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u16.u8 %0, t;}" : "=h"(r) : "f"(a), "f"(b)); o[0]=r;'),
 ("rs_e2m1x4", 'unsigned short r; asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1,%2,%3,%4}, %5;" : "=h"(r) : "f"(a), "f"(b), "f"(a), "f"(b), "r"(0x12345u)); o[0]=r;'),
 ("rn_e4m3x2", 'unsigned short r; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(a), "f"(b)); o[0]=r;'),
]:
    src = 'extern "C" __global__ void k(unsigned* o, float a, float b){' + body + '}'
    try:
        m = Module(src); o = torch.zeros(1, dtype=torch.int32, device="cuda")
        launch(m.fn("k"), (1,), (1,), [o, 1.7, -2.6]); torch.cuda.synchronize()
        print(name, "OK", hex(o.item()))
    except Exception as e:
        print(name, "FAIL", str(e).splitlines()[-2:])
