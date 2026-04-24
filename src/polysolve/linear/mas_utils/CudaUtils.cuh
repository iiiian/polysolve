#pragma once

#include <cuda/stream>
#include <cuda/memory_pool>
#include <cuda/buffer>
#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/optional>

#include <new>
#include <sstream>
#include <stdexcept>
#include <string>

#define __both__ __host__ __device__

// Convenient namespace alias for libcudacxx.
namespace cu = ::cuda;
namespace ctd = ::cuda::std;

namespace polysolve::linear::mas
{
    /// @brief ceil(num/denom)
    constexpr int div_round_up(int num, int denom)
    {
        return (num + denom - 1) / denom;
    }

    struct CudaRuntime
    {
        cu::stream_ref stream;
        cu::device_memory_pool_ref mr;
    };

    class CudaOutOfMemoryError : public std::runtime_error
    {
    public:
        using std::runtime_error::runtime_error;
    };

    inline size_t bytes_to_mb_ceil(size_t bytes)
    {
        constexpr size_t mb = 1024ull * 1024ull;
        return (bytes + mb - 1) / mb;
    }

    inline size_t bytes_to_mb_floor(size_t bytes)
    {
        constexpr size_t mb = 1024ull * 1024ull;
        return bytes / mb;
    }

    inline size_t query_free_device_bytes()
    {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        return free_bytes;
    }

    [[noreturn]] inline void throw_cuda_oom(
        const char *prefix,
        const std::string &context,
        size_t requested_bytes)
    {
        const size_t free_bytes = query_free_device_bytes();

        std::ostringstream ss;
        ss << prefix << " Failed to allocate " << bytes_to_mb_ceil(requested_bytes)
           << " MB for " << context << "; only " << bytes_to_mb_floor(free_bytes)
           << " MB free.";
        throw CudaOutOfMemoryError(ss.str());
    }

    template <class F>
    decltype(auto) with_cuda_oom_context(
        const char *prefix,
        const std::string &context,
        size_t requested_bytes,
        F &&fn)
    {
        try
        {
            return fn();
        }
        catch (const CudaOutOfMemoryError &)
        {
            throw;
        }
        catch (const std::bad_alloc &)
        {
            throw_cuda_oom(prefix, context, requested_bytes);
        }
        catch (const cuda::cuda_error &err)
        {
            if (err.status() == cudaErrorMemoryAllocation)
            {
                throw_cuda_oom(prefix, context, requested_bytes);
            }
            throw;
        }
    }

    /// @brief Transfer device scalar src to host.
    template <typename T>
    T device2host(const T *src, CudaRuntime rt)
    {
        T result;
        cudaMemcpyAsync(&result, src, sizeof(T), cudaMemcpyDeviceToHost,
                        rt.stream.get());
        rt.stream.sync();
        return result;
    }

    /// @brief Transfer host scalar val to device dst. Does not sync stream.
    template <typename T>
    void host2device(T *dst, T val, CudaRuntime rt)
    {
        cudaMemcpyAsync(dst, &val, sizeof(T), cudaMemcpyHostToDevice,
                        rt.stream.get());
    }

    /// @brief Nullable device buffer.
    /// It's very annoying device_buffer does not have default ctor for empty buffer.
    template <typename T>
    using Buf = ctd::optional<cu::device_buffer<T>>;

} // namespace polysolve::linear::mas
