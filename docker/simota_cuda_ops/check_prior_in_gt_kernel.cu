#include "pytorch_cuda_helper.hpp"

template <typename T>
__global__ void check_prior_in_gt_cuda_kernel(
    const T *priors, const T *gt_bboxes, bool *is_in_gts, bool *is_in_cts,
    const int num_prior, const int num_gt, const float center_radius) {

  CUDA_1D_KERNEL_LOOP(index, num_prior * num_gt) {
    int b1 = index / num_gt;
    int b2 = index % num_gt;

    int base1 = b1 * 4;
    T prior_x = priors[base1];
    T prior_y = priors[base1 + 1];
    T prior_x_stride = priors[base1 + 2];
    T prior_y_stride = priors[base1 + 3];

    int base2 = b2 * 4;
    T gt_x1 = gt_bboxes[base2];
    T gt_y1 = gt_bboxes[base2 + 1];
    T gt_x2 = gt_bboxes[base2 + 2];
    T gt_y2 = gt_bboxes[base2 + 3];

    T ct_x = 0.5 * (gt_x1 + gt_x2);
    T ct_y = 0.5 * (gt_y1 + gt_y2);
    T ct_x1 = ct_x - center_radius * prior_x_stride;
    T ct_x2 = ct_x + center_radius * prior_x_stride;
    T ct_y1 = ct_y - center_radius * prior_y_stride;
    T ct_y2 = ct_y + center_radius * prior_y_stride;

    bool is_in_gt = (prior_x > gt_x1) & (prior_x < gt_x2) & (prior_y > gt_y1) &
                    (prior_y < gt_y2);
    bool is_in_ct = (prior_x > ct_x1) & (prior_x < ct_x2) & (prior_y > ct_y1) &
                    (prior_y < ct_y2);

    is_in_gts[index] = is_in_gt;
    is_in_cts[index] = is_in_ct;
  }
}

void CheckPriorsInGtLauncher(Tensor priors, Tensor gt_bboxes, Tensor is_in_gts,
                             Tensor is_in_cts, float center_radius) {
  int output_size = is_in_gts.numel();
  int num_prior = priors.size(0);
  int num_gt = gt_bboxes.size(0);

  at::cuda::CUDAGuard device_guard(priors.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      priors.scalar_type(), "check_prior_in_gt_cuda_kernel", ([&] {
        check_prior_in_gt_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                priors.data_ptr<scalar_t>(), gt_bboxes.data_ptr<scalar_t>(),
                is_in_gts.data_ptr<bool>(), is_in_cts.data_ptr<bool>(),
                num_prior, num_gt, center_radius);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}
