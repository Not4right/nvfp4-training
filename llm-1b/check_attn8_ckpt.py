"""On the real checkpoint + real data: gradient agreement FP8-attention vs cuDNN, relative to FP4's own SR noise."""
import torch
torch.cuda.set_per_process_memory_fraction(0.92)
import torch._inductor.config as ic; ic.use_static_cuda_launcher=False
import model2 as M2
from loader import Mixture
cfg=M2.Config(); B,T=2,2048
m=M2.Model(cfg,B,T); sd=torch.load("ckpt/latest.pt",map_location="cpu",weights_only=False)
m.P.flat.copy_(sd["flat"]); m.amax_use.copy_(sd["amax_use"]); del sd
m.fp4=True; m.acc_scale=1.0; m.refresh_weights()
data=Mixture("data",T); data.pos={k:50_000_000 for k in data.pos}   # far from the current training cursor
x,y=data.batch(B); x=torch.from_numpy(x).cuda(); y=torch.from_numpy(y).cuda(); mask=torch.ones(B,T,device='cuda')
def grads(attn, seed):
    m.attn=attn; torch.manual_seed(seed)
    for _ in range(2): m.microbatch(x,y,mask)          # settle delayed scales (and dS scale)
    m.P.mom.zero_(); m.g_emb.zero_()
    loss=m.microbatch(x,y,mask).item()
    return loss, m.P.mom.cpu().float(), m.g_emb.cpu()
def cos(a,b): return (a.flatten()@b.flatten()/(a.norm()*b.norm())).item()
lc1,mc1,ec1=grads("cudnn",1); lc2,mc2,ec2=grads("cudnn",2); lf,mf,ef=grads("fp8",3)
print(f"loss cudnn {lc1:.4f} / {lc2:.4f}   fp8 {lf:.4f}")
print(f"all block-matrix grads: cos(cudnn,cudnn') {cos(mc1,mc2):.5f}   cos(fp8,cudnn) {cos(mf,mc1):.5f}")
print(f"embedding grads:        cos(cudnn,cudnn') {cos(ec1,ec2):.5f}   cos(fp8,cudnn) {cos(ef,ec1):.5f}")
# per-layer qkv (most affected by attention)
n=sum(a*b for _,(a,b) in m.P.mats); off=0
for l in (0,10,19):
    a,b=m.P.mats[0][1]; k=a*b; base=l*n
    print(f"layer {l} wqkv: cos(cudnn,cudnn') {cos(mc1[base:base+k],mc2[base:base+k]):.5f}  cos(fp8,cudnn) {cos(mf[base:base+k],mc1[base:base+k]):.5f}")
