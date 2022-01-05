#include "pytorch_cpp_helper.hpp"

void check_prior_in_gt(Tensor priors, Tensor gt_bboxes, Tensor is_in_gts,
                        Tensor is_in_cts, float center_radius);

void binary_cross_entropy_cost(Tensor valid_pred_scores, Tensor gt_onehot_label,
                               Tensor cost_matrix);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("check_prior_in_gt", &check_prior_in_gt,
        "check if priors in gt_bboxes or gt center regions");

  m.def("binary_cross_entropy_cost", &binary_cross_entropy_cost,
        "classification cost of SimOTA using binary_cross_entropy loss");
}
