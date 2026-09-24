"""Compile CUDA source with NVRTC for sm_120a and launch on torch's stream.

No nvcc on this machine, so everything goes through NVRTC (the pip wheel)
and is loaded as a cubin through CuPy's driver wrapper.
"""
import hashlib
import os
import numpy as np
import cupy
from cupy.cuda import nvrtc
from cupy.cuda import function as _cpf

_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cubin_cache")
os.makedirs(_CACHE, exist_ok=True)


def compile_cubin(src, arch="sm_120a", opts=()):
    import nvidia.cuda_runtime
    inc = os.path.join(list(nvidia.cuda_runtime.__path__)[0], "include")
    options = [f"--gpu-architecture={arch}", "-std=c++17", "--use_fast_math", "-default-device", f"-I{inc}", *opts]
    key = hashlib.sha1((src + "|".join(options)).encode()).hexdigest()[:16]
    path = os.path.join(_CACHE, key + ".cubin")
    if os.path.exists(path):
        return open(path, "rb").read()
    prog = nvrtc.createProgram(src, "k.cu", [], [])
    try:
        nvrtc.compileProgram(prog, options)
    except Exception as e:
        log = nvrtc.getProgramLog(prog)
        raise RuntimeError(f"NVRTC failed:\n{log}") from e
    cubin = nvrtc.getCUBIN(prog)
    nvrtc.destroyProgram(prog)
    open(path, "wb").write(cubin)
    return cubin


class Module:
    def __init__(self, src, **kw):
        self.mod = _cpf.Module()
        self.mod.load(compile_cubin(src, **kw))
        self._f = {}

    def fn(self, name, smem=0):
        if name not in self._f:
            f = self.mod.get_function(name)
            if smem > 48 * 1024:
                # cudaFuncAttributeMaxDynamicSharedMemorySize = 8
                cupy.cuda.driver.funcSetAttribute(f.ptr, 8, smem)
            self._f[name] = f
        return self._f[name]


def launch(f, grid, block, args, smem=0, stream=None):
    import torch
    if stream is None:
        stream = torch.cuda.current_stream().cuda_stream
    conv = []
    for a in args:
        if hasattr(a, "data_ptr"):
            conv.append(np.uint64(a.data_ptr()))
        elif isinstance(a, float):
            conv.append(np.float32(a))
        elif isinstance(a, int):
            conv.append(np.int32(a) if a < 2**31 else np.uint32(a))
        else:
            conv.append(a)
    f(grid, block, tuple(conv), shared_mem=smem, stream=cupy.cuda.ExternalStream(stream))
