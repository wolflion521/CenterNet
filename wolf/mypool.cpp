#include <torch/torch.h>

#include <vector>

std::vector<at::Tensor> mypool_forward(
    at::Tensor input
) {
    at::Tensor output = at::zeros_like(input);
    return {
        output
    };
}

std::vector<at::Tensor> mypool_backward(
    at::Tensor input,
    at::Tensor grad_output
) {
    auto output = at::zeros_like(input);
    return {
        output
    };
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward", &mypool_forward, "MYPool Forward",
        py::call_guard<py::gil_scoped_release>()
    );
    m.def(
        "backward", &mypool_backward, "MyPool Backward",
        py::call_guard<py::gil_scoped_release>()
    );
}
