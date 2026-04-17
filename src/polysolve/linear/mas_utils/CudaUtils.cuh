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
    // @brief ceil(num/denom)
    constexpr int div_round_upper(int num, int denom)
    {
        return (num + denom - 1) / denom;
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
