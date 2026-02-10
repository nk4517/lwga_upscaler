from typing import Callable

def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda

gradient_aware_upscale_forward = _make_lazy_cuda_func("gradient_aware_upscale_forward")
gradient_aware_upscale_backward = _make_lazy_cuda_func("gradient_aware_upscale_backward")