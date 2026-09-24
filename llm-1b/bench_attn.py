import torch, time
torch.cuda.set_per_process_memory_fraction(0.85)
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
B,H,KV,T,D=2,16,4,2048,128
q=torch.randn(B,H,T,D,device='cuda',dtype=torch.bfloat16,requires_grad=True)
k=torch.randn(B,KV,T,D,device='cuda',dtype=torch.bfloat16,requires_grad=True)
v=torch.randn(B,KV,T,D,device='cuda',dtype=torch.bfloat16,requires_grad=True)
fl=4*B*H*T*T*D/2*3.5
def run(name,f):
    try:
        for i in range(4):
            if i==1: torch.cuda.synchronize(); t=time.perf_counter()
            o=f(); o.sum().backward()
        torch.cuda.synchronize(); dt=(time.perf_counter()-t)/3
        print(f"{name}: {dt*1e3:.2f} ms fwd+bwd  ({fl/dt/1e12:.1f} TF)",flush=True)
    except Exception as e: print(name,"FAIL",repr(e)[:150],flush=True)
rep=lambda x: x.repeat_interleave(H//KV,1)
for be in [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION]:
    def f(be=be):
        with sdpa_kernel(be): return F.scaled_dot_product_attention(q,rep(k),rep(v),is_causal=True)
    run(str(be),f)
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
bm=create_block_mask(lambda b,h,qi,ki: qi>=ki, None,None,T,T)
fa=torch.compile(flex_attention)
run("flex(gqa)", lambda: fa(q,k,v,block_mask=bm,enable_gqa=True))
