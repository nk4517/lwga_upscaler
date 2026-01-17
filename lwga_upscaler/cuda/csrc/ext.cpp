#include "bindings.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gradient_aware_upscale_forward", &gradient_aware_upscale_forward);
    m.def("gradient_aware_upscale_backward", &gradient_aware_upscale_backward);
}
