#pragma once

#include <catch2/catch.hpp>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <polysolve/Types.hpp>
#include <polysolve/linear/mas_utils/CudaUtils.cuh>

#include <cuda/algorithm>
#include <cuda/devices>
#include <cuda/stream>
#include <cuda/memory_pool>
#include <cuda/std/optional>

#include <random>
#include <vector>

namespace polysolve::linear::mas::test
{

    inline std::mt19937 make_rng()
    {
        return std::mt19937(Catch::rngSeed());
    }

    inline StiffnessMatrix make_random_sparse(int n, double density, std::mt19937 &rng)
    {
        std::uniform_real_distribution<double> val_dist(-1.0, 1.0);
        std::uniform_int_distribution<int> idx_dist(0, n - 1);

        int target_nnz = double(n) * n * density;
        if (target_nnz < n)
            target_nnz = n;

        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(target_nnz + n);

        for (int i = 0; i < n; ++i)
        {
            double v = val_dist(rng);
            if (v == 0.0)
                v = 1.0;
            trips.emplace_back(i, i, v);
        }

        for (int k = 0; k < target_nnz; ++k)
        {
            int i = idx_dist(rng);
            int j = idx_dist(rng);
            double v = val_dist(rng);
            if (v == 0.0)
                continue;
            trips.emplace_back(i, j, v);
        }

        StiffnessMatrix A(n, n);
        A.setFromTriplets(trips.begin(), trips.end());
        A.makeCompressed();
        return A;
    }

    inline Eigen::VectorXd make_random_vector(int n, std::mt19937 &rng)
    {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        Eigen::VectorXd v(n);
        for (int i = 0; i < n; ++i)
            v[i] = dist(rng);
        return v;
    }

    struct CudaContext
    {
        ctd::optional<cu::device_ref> device;
        ctd::optional<cu::stream> stream;
        ctd::optional<cu::device_memory_pool> pool;

        CudaContext()
        {
            if (cu::devices.size() == 0)
                throw std::runtime_error("No CUDA device available");
            device.emplace(cu::devices[0]);
            stream.emplace(*device);
            pool.emplace(*device);
        }

        CudaRuntime rt() { return CudaRuntime{*stream, pool->as_ref()}; }
    };

} // namespace polysolve::linear::mas::test
