import torch
torch.cuda.set_per_process_memory_fraction(0.85)
import torch._inductor.config as ic; ic.use_static_cuda_launcher=False
import model as R
torch.manual_seed(0); dev='cuda'
L=2; B=1; T=2048
ref=R.GPT(R.Config(layers=L)).cuda().bfloat16()
with torch.no_grad():
    for b in ref.blocks:
        for p in (b.n1,b.n2,b.qn,b.kn): p.add_(torch.randn_like(p)*0.1)
idx=torch.randint(0,49152,(B,T+1),device=dev); x,y=idx[:,:-1].contiguous(),idx[:,1:].contiguous()
ref.grad_ckpt=False
def grads(fp4,seed=0):
    R.LinCtx.fp4=fp4; R.LinCtx.seed=seed
    for p in ref.parameters(): p.grad=None
    ref(x,y).backward()
    return {n:p.grad.float().clone() for n,p in ref.named_parameters()}
def cos(a,b): a,b=a.flatten(),b.flatten(); return (a@b/(a.norm()*b.norm())).item()
g0=grads(False); g1=grads(True,1); g2=grads(True,2)
for n in g0:
    if "blocks.1" in n or n=="emb":
        print(f"{n:22s} fp4-vs-bf16 {cos(g1[n],g0[n]):.4f}   fp4-vs-fp4(other SR seed) {cos(g1[n],g2[n]):.4f}")
