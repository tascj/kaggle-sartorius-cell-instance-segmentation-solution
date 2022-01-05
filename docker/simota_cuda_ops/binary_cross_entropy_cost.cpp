#include "pytorch_cpp_helper.hpp"

// C++ interface

void BinaryCrossEntropyCostLauncher(Tensor valid_pred_scores,
                                    Tensor gt_onehot_label, Tensor cost_matrix);

void binary_cross_entropy_cost(Tensor valid_pred_scores, Tensor gt_onehot_label,
                               Tensor cost_matrix) {
  CHECK_CUDA_INPUT(valid_pred_scores);
  CHECK_CUDA_INPUT(gt_onehot_label);

  BinaryCrossEntropyCostLauncher(valid_pred_scores, gt_onehot_label,
                                 cost_matrix);
}
