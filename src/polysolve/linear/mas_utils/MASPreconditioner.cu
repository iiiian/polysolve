#include <polysolve/linear/mas_utils/MASPreconditioner.hpp>

#include <polysolve/linear/mas_utils/CudaUtils.cuh>

#include <cub/cub.cuh>
#include <cuda/algorithm>
#include <cuda/atomic>
#include <cuda/std/bit>

#define IDXTYPEWIDTH 32
#define REALTYPEWIDTH 32
#include <metis.h>
#undef IDXTYPEWIDTH
#undef REALTYPEWIDTH

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace polysolve::linear::mas
{
    namespace
    {
        constexpr int BANK_SIZE = 32;
        constexpr int KERNEL_BLOCK_SIZE = 128;
        constexpr int MAX_LEVEL = 6;

        template <int D>
        __global__ void build_multilevel_r_kernel(
            const double *fine_r,
            double *multi_r,
            const int *fine_ancestors,
            const int *level_offsets,
            int max_level_count,
            int level_count,
            int fine_node_count)
        {
            int fine = blockIdx.x * blockDim.x + threadIdx.x;
            if (fine >= fine_node_count)
            {
                return;
            }

            int ancestor_offset = fine * max_level_count;
            for (int d = 0; d < D; ++d)
            {
                double val = fine_r[fine * D + d];
                for (int level = 0; level < level_count; ++level)
                {
                    int node = fine_ancestors[ancestor_offset + level];
                    cu::atomic_ref<double> out{multi_r[(level_offsets[level] + node) * D + d]};
                    out.fetch_add(val, ctd::memory_order_relaxed);
                }
            }
        }

        template <int D>
        __global__ void apply_bank_inverse_kernel(
            const double *inverse_matrices,
            const double *multi_r,
            double *multi_z,
            int node_offset,
            int matrix_offset,
            int bank_count)
        {
            constexpr int N = BANK_SIZE * D;

            int bank = blockIdx.x;
            int row = threadIdx.x;
            if (bank >= bank_count || row >= N)
            {
                return;
            }

            int matrix_base = (matrix_offset + bank) * N * N;
            int vector_base = (node_offset + bank * BANK_SIZE) * D;

            double sum = 0.0;
            for (int col = 0; col < N; ++col)
            {
                sum += inverse_matrices[matrix_base + row * N + col] * multi_r[vector_base + col];
            }
            multi_z[vector_base + row] = sum;
        }

        template <int D>
        __global__ void collect_final_z_kernel(
            const double *multi_z,
            double *fine_z,
            const int *fine_ancestors,
            const int *level_offsets,
            int max_level_count,
            int level_count,
            int fine_node_count)
        {
            int fine = blockIdx.x * blockDim.x + threadIdx.x;
            if (fine >= fine_node_count)
            {
                return;
            }

            int ancestor_offset = fine * max_level_count;
            for (int d = 0; d < D; ++d)
            {
                double sum = 0.0;
                for (int level = 0; level < level_count; ++level)
                {
                    int node = fine_ancestors[ancestor_offset + level];
                    sum += multi_z[(level_offsets[level] + node) * D + d];
                }
                fine_z[fine * D + d] = sum;
            }
        }

        __global__ void store_level0_mapping_kernel(
            const int *level0_map, int *fine_ancestors, int max_level_count, int fine_node_count)
        {
            int fine = blockIdx.x * blockDim.x + threadIdx.x;
            if (fine >= fine_node_count)
            {
                return;
            }

            fine_ancestors[fine * max_level_count] = level0_map[fine];
        }

        __global__ void build_local_connectivity_kernel(
            TopologyView topology,
            const int *fine_ancestors,
            int max_level_count,
            int level,
            uint32_t *connect_masks)
        {
            int fine = blockIdx.x * blockDim.x + threadIdx.x;
            if (fine >= topology.dim)
            {
                return;
            }

            int row_node = fine_ancestors[fine * max_level_count + level];
            if (row_node < 0)
            {
                return;
            }

            int row_bank = row_node / BANK_SIZE;
            for (int idx = topology.row_ptr[fine]; idx < topology.row_ptr[fine + 1]; ++idx)
            {
                int neighbor = topology.cols[idx];
                if (neighbor == fine)
                {
                    continue;
                }

                int col_node = fine_ancestors[neighbor * max_level_count + level];
                if (col_node < 0 || col_node == row_node || (col_node / BANK_SIZE) != row_bank)
                {
                    continue;
                }

                uint32_t bit = uint32_t{1} << (col_node % BANK_SIZE);
                cuda::atomic_ref<uint32_t, cuda::thread_scope_device> out(connect_masks[row_node]);
                out.fetch_or(bit, ctd::memory_order_relaxed);
            }
        }

        __global__ void close_components_and_count_kernel(
            uint32_t *connect_masks,
            int current_node_count,
            const int *level_remap,
            int *component_counts)
        {
            int bank = blockIdx.x;
            int lane = threadIdx.x;
            int node = bank * BANK_SIZE + lane;

            __shared__ uint32_t direct[BANK_SIZE];
            using WarpReduce = cub::WarpReduce<int, BANK_SIZE>;
            __shared__ typename WarpReduce::TempStorage reduce_storage;

            bool active = node < current_node_count;
            if (level_remap != nullptr)
            {
                active = level_remap[node] >= 0;
            }

            if (active)
            {
                direct[lane] = connect_masks[node] | (uint32_t{1} << lane);
            }
            else
            {
                direct[lane] = 0;
            }
            __syncthreads();

            uint32_t component = direct[lane];
            if (active)
            {
                while (true)
                {
                    uint32_t todo = component;
                    bool changed = false;
                    while (todo)
                    {
                        int next = cuda::std::countr_zero(todo);
                        uint32_t expanded = component | direct[next];
                        changed = changed || (expanded != component);
                        component = expanded;
                        todo &= (todo - 1);
                    }
                    if (!changed)
                    {
                        break;
                    }
                }
                connect_masks[node] = component;
            }

            uint32_t lower = (lane == 0) ? 0 : ((uint32_t{1} << lane) - 1);
            int representative = (active && component != 0 && cuda::std::popcount(component & lower) == 0)
                                     ? 1
                                     : 0;

            int total = WarpReduce(reduce_storage).Sum(representative);
            if (lane == 0)
            {
                component_counts[bank] = total;
            }
        }

        __global__ void compute_next_count_kernel(
            const int *bank_offsets, const int *component_counts, int bank_count, int *next_count)
        {
            if (threadIdx.x != 0 || blockIdx.x != 0)
            {
                return;
            }

            if (bank_count == 0)
            {
                next_count[0] = 0;
                return;
            }

            next_count[0] = bank_offsets[bank_count - 1] + component_counts[bank_count - 1];
        }

        __global__ void assign_next_level_kernel(
            const uint32_t *connect_masks,
            const int *bank_offsets,
            int current_node_count,
            const int *level_remap,
            int *current_to_next)
        {
            int bank = blockIdx.x;
            int lane = threadIdx.x;
            int node = bank * BANK_SIZE + lane;

            __shared__ int representative_prefix[BANK_SIZE];
            using WarpScan = cub::WarpScan<int, BANK_SIZE>;
            __shared__ typename WarpScan::TempStorage scan_storage;
            bool active = node < current_node_count;
            if (level_remap != nullptr)
            {
                active = level_remap[node] >= 0;
            }
            uint32_t component = active ? connect_masks[node] : 0;
            uint32_t lower = (lane == 0) ? 0 : ((uint32_t{1} << lane) - 1);
            int representative =
                (active && cuda::std::popcount(component & lower) == 0) ? 1 : 0;

            int prefix = 0;
            WarpScan(scan_storage).ExclusiveSum(representative, prefix);
            if (representative)
            {
                representative_prefix[lane] = prefix;
            }
            __syncthreads();

            if (active)
            {
                int leader = cuda::std::countr_zero(component);
                current_to_next[node] = bank_offsets[bank] + representative_prefix[leader];
            }
        }

        __global__ void compose_fine_ancestors_kernel(
            int *fine_ancestors,
            int max_level_count,
            int level,
            const int *current_to_next,
            int fine_node_count)
        {
            int fine = blockIdx.x * blockDim.x + threadIdx.x;
            if (fine >= fine_node_count)
            {
                return;
            }

            int current = fine_ancestors[fine * max_level_count + level];
            fine_ancestors[fine * max_level_count + level + 1] = current_to_next[current];
        }

        template <int D>
        __global__ void init_level0_ghost_identity_kernel(
            double *local_matrices, const int *level0_remap, int matrix_offset, int bank_count)
        {
            int node = blockIdx.x * blockDim.x + threadIdx.x;
            if (node >= bank_count * BANK_SIZE)
            {
                return;
            }

            if (level0_remap[node] >= 0)
            {
                return;
            }

            constexpr int N = BANK_SIZE * D;
            int bank = node / BANK_SIZE;
            int local = node % BANK_SIZE;
            int base = (matrix_offset + bank) * N * N;
            for (int d = 0; d < D; ++d)
            {
                int row = local * D + d;
                local_matrices[base + row * N + row] = 1.0;
            }
        }

        template <int D>
        __global__ void init_padded_identity_kernel(
            double *local_matrices, int node_count, int matrix_offset, int bank_count)
        {
            int node = blockIdx.x * blockDim.x + threadIdx.x;
            if (node >= bank_count * BANK_SIZE || node < node_count)
            {
                return;
            }

            constexpr int N = BANK_SIZE * D;
            int bank = node / BANK_SIZE;
            int local = node % BANK_SIZE;
            int base = (matrix_offset + bank) * N * N;
            for (int d = 0; d < D; ++d)
            {
                int row = local * D + d;
                local_matrices[base + row * N + row] = 1.0;
            }
        }

        template <int D>
        __global__ void assemble_level_matrices_kernel(
            BCOOView matrix,
            const int *fine_ancestors,
            int max_level_count,
            int level,
            int matrix_offset,
            double *local_matrices)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= matrix.non_zeros)
            {
                return;
            }

            int row = matrix.rows[idx];
            int col = matrix.cols[idx];
            int row_node = fine_ancestors[row * max_level_count + level];
            int col_node = fine_ancestors[col * max_level_count + level];
            if (row_node < 0 || col_node < 0 || row_node / BANK_SIZE != col_node / BANK_SIZE)
            {
                return;
            }

            constexpr int N = BANK_SIZE * D;
            int matrix_base = (matrix_offset + row_node / BANK_SIZE) * N * N;
            int local_row = (row_node % BANK_SIZE) * D;
            int local_col = (col_node % BANK_SIZE) * D;
            int value_base = idx * D * D;
            for (int i = 0; i < D; ++i)
            {
                for (int j = 0; j < D; ++j)
                {
                    cu::atomic_ref<double> out{
                        local_matrices[matrix_base + (local_row + i) * N + local_col + j]};
                    out.fetch_add(matrix.vals[value_base + i * D + j], ctd::memory_order_relaxed);
                }
            }
        }

        template <int D>
        __global__ void assemble_all_levels_matrices_kernel(
            BCOOView matrix,
            const int *fine_ancestors,
            int max_level_count,
            int level_count,
            const int *level_matrix_offsets,
            double *local_matrices)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= matrix.non_zeros)
            {
                return;
            }

            int row = matrix.rows[idx];
            int col = matrix.cols[idx];
            constexpr int N = BANK_SIZE * D;
            int value_base = idx * D * D;

            for (int level = 0; level < level_count; ++level)
            {
                int row_node = fine_ancestors[row * max_level_count + level];
                int col_node = fine_ancestors[col * max_level_count + level];
                if (row_node < 0 || col_node < 0 || row_node / BANK_SIZE != col_node / BANK_SIZE)
                {
                    continue;
                }

                int matrix_base = (level_matrix_offsets[level] + row_node / BANK_SIZE) * N * N;
                int local_row = (row_node % BANK_SIZE) * D;
                int local_col = (col_node % BANK_SIZE) * D;
                for (int i = 0; i < D; ++i)
                {
                    for (int j = 0; j < D; ++j)
                    {
                        cu::atomic_ref<double> out{
                            local_matrices[matrix_base + (local_row + i) * N + local_col + j]};
                        out.fetch_add(matrix.vals[value_base + i * D + j], ctd::memory_order_relaxed);
                    }
                }
            }
        }

        template <int D>
        __global__ void invert_banks_kernel(
            double *local_matrices, double *inverse_matrices, int matrix_offset, int bank_count)
        {
            constexpr int N = BANK_SIZE * D;

            int bank = blockIdx.x;
            int row = threadIdx.x;
            if (bank >= bank_count || row >= N)
            {
                return;
            }

            __shared__ int failed;
            __shared__ int swap_row;
            if (row == 0)
            {
                failed = 0;
                swap_row = 0;
            }
            __syncthreads();

            int base = (matrix_offset + bank) * N * N;
            for (int col = 0; col < N; ++col)
            {
                inverse_matrices[base + row * N + col] = (row == col) ? 1.0 : 0.0;
            }
            __syncthreads();

            for (int pivot = 0; pivot < N; ++pivot)
            {
                if (row == 0)
                {
                    int best_row = pivot;
                    double best_abs = ctd::abs(local_matrices[base + pivot * N + pivot]);
                    for (int candidate = pivot + 1; candidate < N; ++candidate)
                    {
                        double candidate_abs =
                            ctd::abs(local_matrices[base + candidate * N + pivot]);
                        if (candidate_abs > best_abs)
                        {
                            best_abs = candidate_abs;
                            best_row = candidate;
                        }
                    }

                    if (best_abs < 1e-20)
                    {
                        failed = 1;
                    }
                    else
                    {
                        swap_row = best_row;
                    }
                }
                __syncthreads();

                if (failed)
                {
                    break;
                }

                if (swap_row != pivot && (row == pivot || row == swap_row))
                {
                    int other = (row == pivot) ? swap_row : pivot;
                    for (int col = 0; col < N; ++col)
                    {
                        double tmp_a = local_matrices[base + row * N + col];
                        local_matrices[base + row * N + col] =
                            local_matrices[base + other * N + col];
                        local_matrices[base + other * N + col] = tmp_a;

                        double tmp_inv = inverse_matrices[base + row * N + col];
                        inverse_matrices[base + row * N + col] =
                            inverse_matrices[base + other * N + col];
                        inverse_matrices[base + other * N + col] = tmp_inv;
                    }
                }
                __syncthreads();

                double pivot_value = local_matrices[base + pivot * N + pivot];
                if (row == pivot)
                {
                    double inv_pivot = 1.0 / pivot_value;
                    for (int col = 0; col < N; ++col)
                    {
                        local_matrices[base + pivot * N + col] *= inv_pivot;
                        inverse_matrices[base + pivot * N + col] *= inv_pivot;
                    }
                }
                __syncthreads();

                if (row != pivot)
                {
                    double factor = local_matrices[base + row * N + pivot];
                    if (factor != 0.0)
                    {
                        for (int col = 0; col < N; ++col)
                        {
                            local_matrices[base + row * N + col] -=
                                factor * local_matrices[base + pivot * N + col];
                            inverse_matrices[base + row * N + col] -=
                                factor * inverse_matrices[base + pivot * N + col];
                        }
                    }
                }
                __syncthreads();
            }

            if (failed)
            {
                for (int col = 0; col < N; ++col)
                {
                    inverse_matrices[base + row * N + col] = (row == col) ? 1.0 : 0.0;
                }
            }
        }

        void build_undirected_adjacency(
            const std::vector<int> &row_ptr,
            const std::vector<int> &cols,
            std::vector<std::vector<int>> &adjacency)
        {
            int dim = row_ptr.size() - 1;
            adjacency.assign(dim, {});
            for (int row = 0; row < dim; ++row)
            {
                for (int idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx)
                {
                    int col = cols[idx];
                    if (row == col)
                    {
                        continue;
                    }
                    adjacency[row].push_back(col);
                    adjacency[col].push_back(row);
                }
            }

            for (auto &neighbors : adjacency)
            {
                std::sort(neighbors.begin(), neighbors.end());
                neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
            }
        }

        std::vector<int> metis_partition(const std::vector<std::vector<int>> &adjacency, int &part_count_out)
        {
            int node_count = adjacency.size();
            std::vector<int> part(node_count, 0);
            if (node_count == 0)
            {
                part_count_out = 0;
                return part;
            }

            if (node_count <= BANK_SIZE)
            {
                part_count_out = 1;
                return part;
            }

            std::vector<idx_t> xadj(node_count + 1, 0);
            std::vector<idx_t> adjncy;
            for (int i = 0; i < node_count; ++i)
            {
                xadj[i + 1] = xadj[i] + adjacency[i].size();
                for (int neighbor : adjacency[i])
                {
                    adjncy.push_back(neighbor);
                }
            }

            if (adjncy.empty())
            {
                part_count_out = std::max(1, (node_count + BANK_SIZE - 1) / BANK_SIZE);
                for (int i = 0; i < node_count; ++i)
                {
                    part[i] = i / BANK_SIZE;
                }
                return part;
            }

            std::vector<int> part_sizes;
            for (int slack = 0; slack < BANK_SIZE; ++slack)
            {
                int capacity = BANK_SIZE - slack;
                int part_count = std::max(1, (node_count + capacity - 1) / capacity);

                idx_t nvtxs = node_count;
                idx_t ncon = 1;
                idx_t nparts = part_count;
                idx_t objval = 0;
                std::vector<idx_t> metis_part(node_count, 0);
                int ret = METIS_PartGraphKway(
                    &nvtxs, &ncon, xadj.data(), adjncy.data(), nullptr, nullptr, nullptr, &nparts,
                    nullptr, nullptr, nullptr, &objval, metis_part.data());
                if (ret != METIS_OK)
                {
                    throw std::runtime_error("[CudaPCG] METIS partition failed.");
                }

                part_sizes.assign(part_count, 0);
                for (int i = 0; i < node_count; ++i)
                {
                    int p = int(metis_part[i]);
                    if (p < 0 || p >= part_count)
                    {
                        throw std::runtime_error("[CudaPCG] METIS produced invalid partition id.");
                    }
                    part_sizes[p]++;
                }

                int max_size = 0;
                for (int sz : part_sizes)
                {
                    max_size = std::max(max_size, sz);
                }

                if (max_size <= BANK_SIZE)
                {
                    part_count_out = part_count;
                    for (int i = 0; i < node_count; ++i)
                    {
                        part[i] = int(metis_part[i]);
                    }
                    return part;
                }
            }

            throw std::runtime_error("[CudaPCG] METIS partition size constraint failed.");
        }

        void build_level0_map_remap(
            const std::vector<int> &part,
            int part_count,
            std::vector<int> &fine_to_slot,
            std::vector<int> &slot_to_fine,
            int &bank_count)
        {
            int node_count = part.size();
            if (part_count <= 0 && node_count > 0)
            {
                throw std::runtime_error("[CudaPCG] Invalid METIS partition count.");
            }

            std::vector<int> part_sizes(part_count, 0);
            for (int i = 0; i < node_count; ++i)
            {
                int p = part[i];
                if (p < 0 || p >= part_count)
                {
                    throw std::runtime_error("[CudaPCG] Invalid METIS partition id.");
                }
                part_sizes[p]++;
            }

            std::vector<int> part_to_bank(part_count, -1);
            bank_count = 0;
            for (int p = 0; p < part_count; ++p)
            {
                if (part_sizes[p] > 0)
                {
                    part_to_bank[p] = bank_count++;
                }
            }

            fine_to_slot.assign(node_count, -1);
            slot_to_fine.assign(bank_count * BANK_SIZE, -1);
            std::vector<int> bank_cursor(bank_count, 0);
            for (int i = 0; i < node_count; ++i)
            {
                int bank = part_to_bank[part[i]];
                if (bank < 0)
                {
                    continue;
                }
                int local = bank_cursor[bank]++;
                if (local >= BANK_SIZE)
                {
                    throw std::runtime_error("[CudaPCG] METIS partition exceeded bank size.");
                }
                int slot = bank * BANK_SIZE + local;
                fine_to_slot[i] = slot;
                slot_to_fine[slot] = i;
            }
        }
    } // namespace

    bool MASPreconditioner::same_pattern(
        const std::vector<int> &row_ptr, const std::vector<int> &cols) const
    {
        return fine_node_count_ == int(row_ptr.size() - 1) && cached_row_ptr_ == row_ptr
               && cached_cols_ == cols;
    }

    void MASPreconditioner::analyze_pattern(
        TopologyView topology,
        const std::vector<int> &row_ptr,
        const std::vector<int> &cols,
        const std::vector<int> &level0_map,
        const std::vector<int> &level0_remap,
        int level0_bank_count,
        CudaRuntime rt)
    {
        rt.stream.sync();

        cached_row_ptr_ = row_ptr;
        cached_cols_ = cols;
        fine_node_count_ = row_ptr.size() - 1;

        int level0_padded_count = level0_bank_count * BANK_SIZE;
        max_bank_count_ = level0_bank_count;

        fine_ancestors_ =
            cu::make_buffer<int>(rt.stream, rt.mr, fine_node_count_ * MAX_LEVEL, cu::no_init);
        cu::fill_bytes(rt.stream, *fine_ancestors_, ctd::uint8_t{0xff});

        level0_map_ = cu::make_buffer<int>(rt.stream, rt.mr, level0_map.size(), cu::no_init);
        cu::copy_bytes(rt.stream, level0_map, *level0_map_);
        level0_remap_ = cu::make_buffer<int>(rt.stream, rt.mr, level0_remap.size(), cu::no_init);
        cu::copy_bytes(rt.stream, level0_remap, *level0_remap_);
        int fine_grid = compute_grid_num(fine_node_count_, KERNEL_BLOCK_SIZE);
        store_level0_mapping_kernel<<<fine_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
            level0_map_->data(), fine_ancestors_->data(), MAX_LEVEL, fine_node_count_);

        connect_masks_ =
            cu::make_buffer<uint32_t>(rt.stream, rt.mr, level0_padded_count, cu::no_init);
        component_counts_ = cu::make_buffer<int>(rt.stream, rt.mr, max_bank_count_, cu::no_init);
        bank_offsets_ = cu::make_buffer<int>(rt.stream, rt.mr, max_bank_count_, cu::no_init);
        current_to_next_ = cu::make_buffer<int>(rt.stream, rt.mr, level0_padded_count, cu::no_init);
        next_count_ = cu::make_buffer<int>(rt.stream, rt.mr, 1, cu::no_init);

        size_t scan_bytes = 1;
        if (max_bank_count_ > 0)
        {
            cub::DeviceScan::ExclusiveSum(
                nullptr,
                scan_bytes,
                component_counts_->data(),
                bank_offsets_->data(),
                max_bank_count_,
                rt.stream.get());
        }
        scan_storage_ = cu::make_buffer<char>(rt.stream, rt.mr, scan_bytes, cu::no_init);

        levels_.clear();
        level_node_offsets_.clear();
        level_matrix_offsets_.clear();
        levels_.push_back(LevelInfo{level0_padded_count, level0_padded_count, level0_bank_count});

        int current_node_count = level0_padded_count;
        int current_bank_count = level0_bank_count;
        int current_level = 0;
        while (current_level + 1 < MAX_LEVEL && current_node_count > 1)
        {
            cudaMemsetAsync(
                connect_masks_->data(),
                0,
                current_bank_count * BANK_SIZE * sizeof(uint32_t),
                rt.stream.get());

            build_local_connectivity_kernel<<<fine_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
                topology,
                fine_ancestors_->data(),
                MAX_LEVEL,
                current_level,
                connect_masks_->data());

            close_components_and_count_kernel<<<current_bank_count, BANK_SIZE, 0, rt.stream.get()>>>(
                connect_masks_->data(),
                current_node_count,
                current_level == 0 ? level0_remap_->data() : nullptr,
                component_counts_->data());

            size_t scan_bytes = scan_storage_->size();
            cub::DeviceScan::ExclusiveSum(
                scan_storage_->data(),
                scan_bytes,
                component_counts_->data(),
                bank_offsets_->data(),
                current_bank_count,
                rt.stream.get());
            compute_next_count_kernel<<<1, 1, 0, rt.stream.get()>>>(
                bank_offsets_->data(), component_counts_->data(), current_bank_count, next_count_->data());

            int next_node_count = device2host(next_count_->data(), rt);
            if (next_node_count >= current_node_count)
            {
                break;
            }

            assign_next_level_kernel<<<current_bank_count, BANK_SIZE, 0, rt.stream.get()>>>(
                connect_masks_->data(),
                bank_offsets_->data(),
                current_node_count,
                current_level == 0 ? level0_remap_->data() : nullptr,
                current_to_next_->data());
            compose_fine_ancestors_kernel<<<fine_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
                fine_ancestors_->data(),
                MAX_LEVEL,
                current_level,
                current_to_next_->data(),
                fine_node_count_);

            int next_bank_count = (next_node_count + BANK_SIZE - 1) / BANK_SIZE;
            levels_.push_back(LevelInfo{next_node_count, next_bank_count * BANK_SIZE, next_bank_count});
            current_node_count = next_node_count;
            current_bank_count = next_bank_count;
            current_level++;
        }

        level_count_ = levels_.size();
        total_padded_nodes_ = 0;
        int total_bank_count = 0;
        for (auto &level : levels_)
        {
            level_node_offsets_.push_back(total_padded_nodes_);
            total_padded_nodes_ += level.padded_count;
            level_matrix_offsets_.push_back(total_bank_count);
            total_bank_count += level.bank_count;
        }

        level_node_offsets_device_ =
            cu::make_buffer<int>(rt.stream, rt.mr, level_node_offsets_.size(), cu::no_init);
        cu::copy_bytes(rt.stream, level_node_offsets_, *level_node_offsets_device_);
        level_matrix_offsets_device_ =
            cu::make_buffer<int>(rt.stream, rt.mr, level_matrix_offsets_.size(), cu::no_init);
        cu::copy_bytes(rt.stream, level_matrix_offsets_, *level_matrix_offsets_device_);

        int total_matrix_size = 0;
        for (auto &level : levels_)
        {
            total_matrix_size += level.bank_count;
        }
        int local_dim = BANK_SIZE * block_dim_;
        int matrix_entries = total_matrix_size * local_dim * local_dim;
        local_matrices_ = cu::make_buffer<double>(rt.stream, rt.mr, matrix_entries, cu::no_init);
        inverse_matrices_ = cu::make_buffer<double>(rt.stream, rt.mr, matrix_entries, cu::no_init);
        multi_level_r_ =
            cu::make_buffer<double>(rt.stream, rt.mr, total_padded_nodes_ * block_dim_, cu::no_init);
        multi_level_z_ =
            cu::make_buffer<double>(rt.stream, rt.mr, total_padded_nodes_ * block_dim_, cu::no_init);
        rt.stream.sync();
    }

    void MASPreconditioner::factorize(
        const StiffnessMatrix &, const BCOOMatrix &A, CudaRuntime rt)
    {
        int old_block_dim = block_dim_;
        block_dim_ = A.block_dim;

        TopologyView h_topo = A.host_topology_view();

        if (old_block_dim != block_dim_ || !same_pattern(h_topo.row_ptr, h_topo.cols))
        {
            std::vector<std::vector<int>> adjacency;
            build_undirected_adjacency(topology_row_ptr, topology_cols, adjacency);
            int part_count = 0;
            std::vector<int> part = metis_partition(adjacency, part_count);
            std::vector<int> level0_map;
            std::vector<int> level0_remap;
            int bank_count = 0;
            build_level0_map_remap(part, part_count, level0_map, level0_remap, bank_count);
            analyze_pattern(topology, topology_row_ptr, topology_cols, level0_map, level0_remap, bank_count, rt);
        }

        cudaMemsetAsync(
            local_matrices_->data(),
            0,
            local_matrices_->size() * sizeof(double),
            rt.stream.get());

        int matrix_grid = compute_grid_num(A.non_zeros, KERNEL_BLOCK_SIZE);
        for (int current_level = 0; current_level < level_count_; ++current_level)
        {
            auto &level = levels_[current_level];
            int init_grid = compute_grid_num(level.padded_count, KERNEL_BLOCK_SIZE);
            switch (block_dim_)
            {
            case 1:
                if (current_level == 0)
                {
                    init_level0_ghost_identity_kernel<1>
                        <<<init_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
                            local_matrices_->data(), level0_remap_->data(), level_matrix_offsets_[0],
                            level.bank_count);
                }
                else
                {
                    init_padded_identity_kernel<1><<<init_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
                        local_matrices_->data(),
                        level.node_count,
                        level_matrix_offsets_[current_level],
                        level.bank_count);
                }
                break;
            case 2:
                if (current_level == 0)
                {
                    init_level0_ghost_identity_kernel<2>
                        <<<init_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
                            local_matrices_->data(), level0_remap_->data(), level_matrix_offsets_[0],
                            level.bank_count);
                }
                else
                {
                    init_padded_identity_kernel<2><<<init_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
                        local_matrices_->data(),
                        level.node_count,
                        level_matrix_offsets_[current_level],
                        level.bank_count);
                }
                break;
            case 3:
                if (current_level == 0)
                {
                    init_level0_ghost_identity_kernel<3>
                        <<<init_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
                            local_matrices_->data(), level0_remap_->data(), level_matrix_offsets_[0],
                            level.bank_count);
                }
                else
                {
                    init_padded_identity_kernel<3><<<init_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
                        local_matrices_->data(),
                        level.node_count,
                        level_matrix_offsets_[current_level],
                        level.bank_count);
                }
                break;
            default:
                throw std::runtime_error("[CudaPCG] Unsupported block size.");
            }
        }

        switch (block_dim_)
        {
        case 1:
            assemble_all_levels_matrices_kernel<1><<<matrix_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
                A,
                fine_ancestors_->data(),
                MAX_LEVEL,
                level_count_,
                level_matrix_offsets_device_->data(),
                local_matrices_->data());
            break;
        case 2:
            assemble_all_levels_matrices_kernel<2><<<matrix_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
                A,
                fine_ancestors_->data(),
                MAX_LEVEL,
                level_count_,
                level_matrix_offsets_device_->data(),
                local_matrices_->data());
            break;
        case 3:
            assemble_all_levels_matrices_kernel<3><<<matrix_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
                A,
                fine_ancestors_->data(),
                MAX_LEVEL,
                level_count_,
                level_matrix_offsets_device_->data(),
                local_matrices_->data());
            break;
        default:
            throw std::runtime_error("[CudaPCG] Unsupported block size.");
        }

        for (int current_level = 0; current_level < level_count_; ++current_level)
        {
            auto &level = levels_[current_level];
            switch (block_dim_)
            {
            case 1:
                invert_banks_kernel<1><<<level.bank_count, BANK_SIZE, 0, rt.stream.get()>>>(
                    local_matrices_->data(),
                    inverse_matrices_->data(),
                    level_matrix_offsets_[current_level],
                    level.bank_count);
                break;
            case 2:
                invert_banks_kernel<2><<<level.bank_count, BANK_SIZE * 2, 0, rt.stream.get()>>>(
                    local_matrices_->data(),
                    inverse_matrices_->data(),
                    level_matrix_offsets_[current_level],
                    level.bank_count);
                break;
            case 3:
                invert_banks_kernel<3><<<level.bank_count, BANK_SIZE * 3, 0, rt.stream.get()>>>(
                    local_matrices_->data(),
                    inverse_matrices_->data(),
                    level_matrix_offsets_[current_level],
                    level.bank_count);
                break;
            default:
                throw std::runtime_error("[CudaPCG] Unsupported block size.");
            }
        }

        rt.stream.sync();
    }

    void MASPreconditioner::apply(ctd::span<const double> r, ctd::span<double> z, CudaRuntime rt)
    {
        if (empty())
        {
            throw std::runtime_error("[CudaPCG] MAS preconditioner is not initialized.");
        }

        cudaMemsetAsync(
            multi_level_r_->data(), 0, multi_level_r_->size() * sizeof(double), rt.stream.get());

        int fine_grid = compute_grid_num(fine_node_count_, KERNEL_BLOCK_SIZE);
        switch (block_dim_)
        {
        case 1:
            build_multilevel_r_kernel<1><<<fine_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
                r.data(),
                multi_level_r_->data(),
                fine_ancestors_->data(),
                level_node_offsets_device_->data(),
                MAX_LEVEL,
                level_count_,
                fine_node_count_);
            break;
        case 2:
            build_multilevel_r_kernel<2><<<fine_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
                r.data(),
                multi_level_r_->data(),
                fine_ancestors_->data(),
                level_node_offsets_device_->data(),
                MAX_LEVEL,
                level_count_,
                fine_node_count_);
            break;
        case 3:
            build_multilevel_r_kernel<3><<<fine_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
                r.data(),
                multi_level_r_->data(),
                fine_ancestors_->data(),
                level_node_offsets_device_->data(),
                MAX_LEVEL,
                level_count_,
                fine_node_count_);
            break;
        default:
            throw std::runtime_error("[CudaPCG] Unsupported block size.");
        }

        int current_level = 0;
        for (auto &level : levels_)
        {
            switch (block_dim_)
            {
            case 1:
                apply_bank_inverse_kernel<1><<<level.bank_count, BANK_SIZE, 0, rt.stream.get()>>>(
                    inverse_matrices_->data(),
                    multi_level_r_->data(),
                    multi_level_z_->data(),
                    level_node_offsets_[current_level],
                    level_matrix_offsets_[current_level],
                    level.bank_count);
                break;
            case 2:
                apply_bank_inverse_kernel<2><<<level.bank_count, BANK_SIZE * 2, 0, rt.stream.get()>>>(
                    inverse_matrices_->data(),
                    multi_level_r_->data(),
                    multi_level_z_->data(),
                    level_node_offsets_[current_level],
                    level_matrix_offsets_[current_level],
                    level.bank_count);
                break;
            case 3:
                apply_bank_inverse_kernel<3><<<level.bank_count, BANK_SIZE * 3, 0, rt.stream.get()>>>(
                    inverse_matrices_->data(),
                    multi_level_r_->data(),
                    multi_level_z_->data(),
                    level_node_offsets_[current_level],
                    level_matrix_offsets_[current_level],
                    level.bank_count);
                break;
            default:
                throw std::runtime_error("[CudaPCG] Unsupported block size.");
            }
            current_level++;
        }

        switch (block_dim_)
        {
        case 1:
            collect_final_z_kernel<1><<<fine_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
                multi_level_z_->data(),
                z.data(),
                fine_ancestors_->data(),
                level_node_offsets_device_->data(),
                MAX_LEVEL,
                level_count_,
                fine_node_count_);
            break;
        case 2:
            collect_final_z_kernel<2><<<fine_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
                multi_level_z_->data(),
                z.data(),
                fine_ancestors_->data(),
                level_node_offsets_device_->data(),
                MAX_LEVEL,
                level_count_,
                fine_node_count_);
            break;
        case 3:
            collect_final_z_kernel<3><<<fine_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
                multi_level_z_->data(),
                z.data(),
                fine_ancestors_->data(),
                level_node_offsets_device_->data(),
                MAX_LEVEL,
                level_count_,
                fine_node_count_);
            break;
        default:
            throw std::runtime_error("[CudaPCG] Unsupported block size.");
        }
    }

} // namespace polysolve::linear::mas
