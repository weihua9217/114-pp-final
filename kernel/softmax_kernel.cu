#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

// ============================================
// Softmax CUDA Kernel
// softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
// ============================================

template <typename scalar_t>
__global__ void softmax_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_cols
) {
    const int row_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    if (row_idx >= batch_size) return;

    const scalar_t* row_input = input + row_idx * num_cols;
    scalar_t* row_output = output + row_idx * num_cols;

    extern __shared__ char shared_mem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_mem);
    scalar_t thread_max = -FLT_MAX;
    for (int i = tid; i < num_cols; i += block_size) {
        thread_max = fmaxf(thread_max, row_input[i]);
    }
    shared_data[tid] = thread_max;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }
    scalar_t row_max = shared_data[0];
    __syncthreads();

    scalar_t thread_sum = 0.0f;
    for (int i = tid; i < num_cols; i += block_size) {
        scalar_t exp_val = expf(row_input[i] - row_max);
        row_output[i] = exp_val;
        thread_sum += exp_val;
    }
    shared_data[tid] = thread_sum;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    scalar_t row_sum = shared_data[0];
    __syncthreads();

    for (int i = tid; i < num_cols; i += block_size) {
        row_output[i] = row_output[i] / row_sum;
    }
}

torch::Tensor softmax_cuda_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor");

    const int batch_size = input.size(0);
    const int num_cols = input.size(1);

    auto output = torch::empty_like(input);
    const int threads_per_block = 256;
    const int num_blocks = batch_size;
    const int shared_mem_size = threads_per_block * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softmax_forward_cuda", ([&] {
        softmax_forward_kernel<scalar_t><<<num_blocks, threads_per_block, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            num_cols
        );
    }));
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softmax_cuda_forward, "Softmax forward (CUDA)");
}
