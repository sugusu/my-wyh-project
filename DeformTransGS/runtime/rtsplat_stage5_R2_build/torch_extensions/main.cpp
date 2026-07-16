#include <torch/extension.h>
int answer(){return 42;}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("answer", torch::wrap_pybind_function(answer), "answer");
}