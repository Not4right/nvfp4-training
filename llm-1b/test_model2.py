import torch, sys
torch.cuda.set_per_process_memory_fraction(0.85)
import torch._inductor.config as ic; ic.use_static_cuda_launcher=False
import model as R
import model2 as M2
torch.manual_seed(0); dev='cuda'
L=2; B=1; T=2048
cfg=M2.Config(layers=L); m=M2.Model(cfg,B,T); m.P.init(1)
# perturb norms so their grads are nontrivial
with torch.no_grad():
    for k,v in m.P.small(): v.add_(torch.randn_like(v.float()).bfloat16()*0.1)
ref=R.GPT(R.Config(layers=L)).cuda().bfloat16()
with torch.no_grad():
    ref.emb.copy_(m.P.emb); ref.nf.copy_(m.P.nf)
    for b,d in zip(ref.blocks,m.P.L):
        b.qkv.weight.copy_(d["wqkv"]); b.o.weight.copy_(d["wo"]); b.gu.weight.copy_(d["wgu"]); b.down.weight.copy_(d["wdown"])
        b.n1.copy_(d["n1"]); b.n2.copy_(d["n2"]); b.qn.copy_(d["qn"]); b.kn.copy_(d["kn"])
idx=torch.randint(0,cfg.vocab,(B,T+1),device=dev); x,y=idx[:,:-1].contiguous(),idx[:,1:].contiguous()
mask=torch.ones(B,T,device=dev)
R.LinCtx.fp4=False; ref.grad_ckpt=False
lr=ref(x,y); lr.backward()
def cos(a,b): a,b=a.float().flatten(),b.float().flatten(); return (a@b/(a.norm()*b.norm())).item()
def run(fp4):
    m.fp4=fp4; m.mom.zero_() if hasattr(m,'mom') else None
    m.P.mom.zero_(); m.g_emb.zero_(); [v.zero_() for v in m.g_small.values()]
    m.refresh_weights()
    if fp4:  # calibrate delayed scales
        for _ in range(2):
            m.microbatch(x,y,mask)
        m.P.mom.zero_(); m.g_emb.zero_(); [v.zero_() for v in m.g_small.values()]
    loss=m.microbatch(x,y,mask).item()
    print(f"--- fp4={fp4}: loss {loss:.5f} vs ref {lr.item():.5f}")
    print(f"  emb cos {cos(m.g_emb, ref.emb.grad):.5f}   nf cos {cos(m.g_small['nf'], ref.nf.grad):.4f}")
    for l,(b,d) in enumerate(zip(ref.blocks,m.P.L)):
        s="  L%d "%l
        for k,rw in (("wqkv",b.qkv.weight),("wo",b.o.weight),("wgu",b.gu.weight),("wdown",b.down.weight)):
            s+=f"{k} {cos(d['m_'+k], rw.grad):.4f}  "
        for k,rw in (("n1",b.n1),("n2",b.n2),("qn",b.qn),("kn",b.kn)):
            s+=f"{k} {cos(m.g_small[k+str(l)], rw.grad):.3f} "
        print(s)
run(False); run(True)
