#include "pytorch_cpp_helper.hpp"

void CheckPriorsInGtLauncher(Tensor priors, Tensor gt_bboxes, Tensor is_in_gts,
                             Tensor is_in_cts, float center_radius);

void check_prior_in_gt(Tensor priors, Tensor gt_bboxes, Tensor is_in_gts,
                        Tensor is_in_cts, float center_radius) {
  CHECK_CUDA_INPUT(priors);
  CHECK_CUDA_INPUT(gt_bboxes);
  CHECK_CUDA_INPUT(is_in_gts);
  CHECK_CUDA_INPUT(is_in_cts);

  return CheckPriorsInGtLauncher(priors, gt_bboxes, is_in_gts, is_in_cts,
                                 center_radius);
}
