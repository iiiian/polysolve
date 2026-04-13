#pragma once

#include <cuda/stream>
#include <cuda/memory_pool>
#include <cuda/buffer>
#include <cuda/std/optional>

#define __both__ __host__ __device__

namespace cu = ::cuda;
namespace ctd = ::cuda::std;

namespace polysolve::linear::mas
{
    /// @brief Compute grid num given minimal thread num and a fixed block dim.
    constexpr int compute_grid_num(int thread_num, int block_dim)
    {
        return (thread_num + block_dim - 1) / block_dim;
    }

    struct CudaRuntime
    {
        cu::stream_ref stream;
        cu::device_memory_pool_ref mr;
    };

    template <typename T>
    T device2host(const T *src, CudaRuntime rt)
    {
        T result;
        cudaMemcpyAsync(&result, src, sizeof(T), cudaMemcpyDeviceToHost,
                        rt.stream.get());
        rt.stream.sync();
        return result;
    }

    template <typename T>
    void host2device(T *dst, T val, CudaRuntime rt)
    {
        cudaMemcpyAsync(dst, &val, sizeof(T), cudaMemcpyHostToDevice,
                        rt.stream.get());
    }

    template <typename T>
    using Buf = ctd::optional<cu::device_buffer<T>>;

} // namespace polysolve::linear::mas
