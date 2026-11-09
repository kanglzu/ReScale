#include <torch/extension.h>

std::vector<torch::Tensor> crop_resized_cuda(torch::Tensor image, torch::Tensor mask,
    torch::Tensor boxes, torch::Tensor batch_indices, torch::Tensor black,
    std::vector<int64_t> output_size, torch::Tensor output_like, bool return_valid);
std::vector<torch::Tensor> paste_probabilities_cuda(torch::Tensor base, torch::Tensor crop,
    torch::Tensor boxes, torch::Tensor sizes, std::vector<int64_t> canvas_size);
torch::Tensor component_stats_cuda(torch::Tensor labels);
std::vector<torch::Tensor> component_boxes_cuda(torch::Tensor stats, torch::Tensor labels,
    torch::Tensor roots, double margin);
void accumulate_crop_probabilities_cuda(torch::Tensor crops, torch::Tensor boxes,
    torch::Tensor indices, std::vector<int64_t> image_size, torch::Tensor sizes,
    torch::Tensor sums, torch::Tensor counts);
torch::Tensor finalize_crop_probabilities_cuda(torch::Tensor sums, torch::Tensor counts,
    torch::Tensor sizes, torch::Tensor base, bool return_coverage);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("crop_resized", &crop_resized_cuda, "Fused indexed normalized-image crop and categorical target (CUDA)",
          pybind11::arg("image"), pybind11::arg("mask"), pybind11::arg("boxes"),
          pybind11::arg("batch_indices"), pybind11::arg("black"), pybind11::arg("output_size"),
          pybind11::arg("output_like"), pybind11::arg("return_valid") = true);
    m.def("paste_probabilities", &paste_probabilities_cuda, "Conflict-free single-ROI probability paste (CUDA)");
    m.def("component_stats", &component_stats_cuda, "Class-aware eight-connected component statistics (CUDA)");
    m.def("component_boxes", &component_boxes_cuda, "Separate width/height expanded component boxes (CUDA)");
    m.def("accumulate_crop_probabilities", &accumulate_crop_probabilities_cuda, "Indexed ROI probability sum/count (CUDA)");
    m.def("finalize_crop_probabilities", &finalize_crop_probabilities_cuda, "In-place ROI probability finalization and outside background (CUDA)");
}
