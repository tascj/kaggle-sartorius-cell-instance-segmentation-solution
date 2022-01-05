#include "pytorch_cuda_helper.hpp"

template <typename T>
__global__ void binary_cross_entropy_cost_cuda_kernel(
    const T *valid_pred_scores, const T *gt_onehot_label, T *cost_matrix,
    const int num_pred_scores, const int num_gt, const int num_classes) {

  CUDA_1D_KERNEL_LOOP(index, num_pred_scores * num_gt) {
    int b1 = index / num_gt;
    int b2 = index % num_gt;

    int base1 = b1 * num_classes;
    int base2 = b2 * num_classes;

    // https://github.com/pytorch/pytorch/blob/v1.10.0/aten/src/ATen/native/cuda/Loss.cu#L100-L112
    const T zero = 0;
    const T one = 1;
    const T neg_100 = -100;
    T cost = 0;
    for (int offset = 0; offset < num_classes; offset++) {
      T input_val = valid_pred_scores[base1 + offset];
      T target_val = gt_onehot_label[base2 + offset];

      CUDA_KERNEL_ASSERT(input_val >= zero && input_val <= one)

      T log_input_val = std::log(input_val);
      T log_1_minus_input_val = std::log(one - input_val);

      log_input_val = std::max(log_input_val, neg_100);
      log_1_minus_input_val = std::max(log_1_minus_input_val, neg_100);

      cost += ((target_val - one) * log_1_minus_input_val -
               (target_val * log_input_val));
    }
    cost_matrix[index] = cost;
  }
}

void BinaryCrossEntropyCostLauncher(Tensor valid_pred_scores,
                                    Tensor gt_onehot_label,
                                    Tensor cost_matrix) {
  int num_pred_scores = valid_pred_scores.size(0);
  int num_gt = gt_onehot_label.size(0);
  int num_classes = valid_pred_scores.size(1);
  int output_size = cost_matrix.numel();

  at::cuda::CUDAGuard device_guard(valid_pred_scores.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      valid_pred_scores.scalar_type(), "binary_cross_entropy_cost_cuda_kernel",
      ([&] {
        binary_cross_entropy_cost_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                valid_pred_scores.data_ptr<scalar_t>(),
                gt_onehot_label.data_ptr<scalar_t>(),
                cost_matrix.data_ptr<scalar_t>(), num_pred_scores, num_gt,
                num_classes);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}
