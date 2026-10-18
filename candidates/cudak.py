"""Hand-written CUDA kernels without a C++ extension: compiled at runtime with NVRTC (the nvidia-cuda-nvrtc wheel that
ships with PyTorch, so no system CUDA toolkit or host compiler is needed), loaded and launched through the CUDA driver
API (cuda-python) on PyTorch's current stream. Compiled images are cached per (source, device arch) in memory and on
disk (candidates/cuk/build)."""
import ctypes
import hashlib
from pathlib import Path

import torch
from cuda.bindings import driver as cu
from cuda.bindings import nvrtc

_BUILD = Path(__file__).resolve().parent / "cuk" / "build"
_mods = {}


def _check(res, what):
    err = res[0] if isinstance(res, tuple) else res
    if int(err) != 0:
        raise RuntimeError(f"{what} failed: {err}")
    return res[1:] if isinstance(res, tuple) and len(res) > 2 else (res[1] if isinstance(res, tuple) and len(res) == 2 else None)


def _compile(src, name, arch, defines=()):
    prog = _check(nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], []), "nvrtcCreateProgram")
    opts = [f"--gpu-architecture=sm_{arch}".encode(), b"--std=c++17", b"-default-device"]
    opts += [f"-D{d}".encode() for d in defines]
    res = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if int(res[0]) != 0:
        n = _check(nvrtc.nvrtcGetProgramLogSize(prog), "log size")
        log = b" " * n
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"NVRTC compile of {name} failed:\n{log.decode(errors='replace')}")
    n = _check(nvrtc.nvrtcGetCUBINSize(prog), "cubin size")
    img = b" " * n
    _check(nvrtc.nvrtcGetCUBIN(prog, img), "get cubin")
    return img


def module(src_path, defines=()):
    """load (compiling if needed) the CUDA source file; returns a cached CUmodule"""
    torch.cuda.init()
    torch.zeros(1, device="cuda")                       # make the primary context current
    dev = torch.cuda.current_device()
    maj, mnr = torch.cuda.get_device_capability(dev)
    arch = f"{maj}{mnr}"
    src = Path(src_path).read_text()
    key = hashlib.sha1((src + arch + repr(tuple(defines))).encode()).hexdigest()[:16]
    if (key, dev) in _mods:
        return _mods[(key, dev)]
    _BUILD.mkdir(parents=True, exist_ok=True)
    f = _BUILD / f"{Path(src_path).stem}_{arch}_{key}.cubin"
    if f.exists():
        img = f.read_bytes()
    else:
        img = _compile(src, Path(src_path).name, arch, defines)
        f.write_bytes(img)
    mod = _check(cu.cuModuleLoadData(img), "cuModuleLoadData")
    _mods[(key, dev)] = mod
    return mod


class Kernel:
    def __init__(self, src_path, name, argtypes, defines=()):
        self.src, self.name, self.argtypes, self.defines = src_path, name, tuple(argtypes), tuple(defines)
        self._fn = {}

    def fn(self):
        dev = torch.cuda.current_device()
        if dev not in self._fn:
            self._fn[dev] = _check(cu.cuModuleGetFunction(module(self.src, self.defines), self.name.encode()), self.name)
        return self._fn[dev]

    def set_smem(self, nbytes):
        _check(cu.cuFuncSetAttribute(self.fn(), cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                     nbytes), "set smem")

    def __call__(self, grid, block, args, smem=0, cooperative=False):
        vals = tuple(a.data_ptr() if torch.is_tensor(a) else a for a in args)
        s = cu.CUstream(torch.cuda.current_stream().cuda_stream)
        launch = cu.cuLaunchCooperativeKernel if cooperative else cu.cuLaunchKernel
        g = tuple(grid) + (1,) * (3 - len(grid))
        b = tuple(block) + (1,) * (3 - len(block))
        if cooperative:
            res = launch(self.fn(), *g, *b, smem, s, (vals, self.argtypes))
        else:
            res = launch(self.fn(), *g, *b, smem, s, (vals, self.argtypes), 0)
        if int(res[0]) != 0:
            raise RuntimeError(f"launch {self.name} failed: {res[0]}")


def num_sms():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


P = ctypes.c_void_p
I32 = ctypes.c_int
F32 = ctypes.c_float
F64 = ctypes.c_double
