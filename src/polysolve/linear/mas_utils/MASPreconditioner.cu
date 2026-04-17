#include <polysolve/linear/mas_utils/MASPreconditioner.hpp>

#include <polysolve/linear/mas_utils/CudaUtils.cuh>
#include <polysolve/linear/mas_utils/MetisPartition.hpp>

#include <cub/cub.cuh>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/atomic>
#include <cuda/warp>
#include <cuda/std/bit>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace polysolve::linear::mas
{
    namespace
    {
        constexpr int MAX_COARSE_LEVEL = 6;

        /// @brief Get coarse space CCO id.
        /// @param map Coarse space map.
        /// @param vid Vertex id.
        /// @param level Coarse space level. Start at lv 0.
        /// @return CCO id.
        __both__ int get_coarse_space_id(ctd::span<const int> map, int vid, int level)
        {
            assert(level >= 0);
            return map[vid * MAX_COARSE_LEVEL + level];
        }

        /// @brief Build local CCO mapping from input space -> coarse space lv 0 and collapse topology.
        /// @param row_ptr CSR graph topology (does not include self).
        /// @param cols CSR graph topology (does not include self).
        /// @param cco_num_per_bank Independent CCO number per bank.
        /// @param local_cco_ids Bank local CCO id at coarse space lv 0.
        __global__ void build_local_cco_lv0(ctd::span<const int> row_ptr,
                                            ctd::span<int> cols,
                                            ctd::span<int> cco_num_per_bank,
                                            ctd::span<int> local_cco_ids)
        {
            // Bank local neighbor masks.
            __shared__ uint32_t neighbors[128];

            int btid = threadIdx.x;                   // block local thread id
            int tid = blockDim.x * blockIdx.x + btid; // global thread id
            int wid = threadIdx.x / 32;               // block local warp id
            int lid = threadIdx.x % 32;               // lane id
            int bid = tid / 32;                       // bank id

            int node_num = row_ptr.size() - 1;
            if (tid >= node_num)
            {
                return;
            }

            // Build neighbor mask (including self) and collapse topology.
            uint32_t neighbor = (1u << lid);
            int out_of_bank_neighbor_count = 0;
            for (int n = row_ptr[tid]; n < row_ptr[tid + 1]; ++n)
            {
                int neighbor_vid = cols[n];
                int neighbor_bid = neighbor_vid / 32;
                if (bid == neighbor_bid)
                {
                    neighbor |= (1u << (neighbor_vid % 32));
                }
                else
                {
                    // Compact topology to include out of bank neighbors exclusively. So we dont
                    // need additional filtering when building future coarse spaces.
                    cols[out_of_bank_neighbor_count] = neighbor_vid;
                    ++out_of_bank_neighbor_count;
                }
            }
            if (out_of_bank_neighbor_count != (row_ptr[tid + 1] - row_ptr[tid]))
            {
                // -1 denotes neighbor list ending.
                cols[out_of_bank_neighbor_count] = -1;
            }
            neighbors[btid] = neighbor;
            __syncwarp();

            // Build connectivity mask (including self).
            uint32_t connection = neighbor;
            uint32_t visited = (1u << lid);
            uint32_t to_visit = connection ^ visited;
            while (to_visit)
            {
                int visiting = ctd::countr_one(to_visit);
                connection |= neighbors[visiting + wid * 32];
                visited |= (1u << visiting);
                to_visit = connection ^ visited;
            }

            // Find bank (warp) local connected component (CCO).
            uint32_t cco_lead_lid = ctd::countr_one(connection);
            bool is_lead = (cco_lead_lid == lid);
            uint32_t lead_lanes = __ballot_sync(0xFFFFFFFFu, is_lead);
            uint32_t before_cco_lead_mask = static_cast<uint32_t>((1ull << cco_lead_lid) - 1ull);

            // Write cco info back.
            local_cco_ids[tid] = ctd::popcount(lead_lanes | before_cco_lead_mask);
            if (lid == 0)
            {
                int bank_cco_num = ctd::popcount(lead_lanes);
                cco_num_per_bank[bid] = bank_cco_num;
            }
        }

        /// @brief Build bank neighbor mask and collapse topology.
        /// @param level Target coarse space level.
        /// @param coarse_space_map Coarse space map.
        /// @param row_ptr CSR graph topology (does not include self).
        /// @param cols CSR graph topology (does not include self).
        /// @param neighbors Bank local neighbor mask.
        __global__ void build_neighbor_masks_lvx(int level,
                                                 ctd::span<const int> coarse_space_map,
                                                 ctd::span<const int> row_ptr,
                                                 ctd::span<int> cols,
                                                 ctd::span<uint32_t> neighbors)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x; // global thread id
            int node_num = row_ptr.size() - 1;
            if (tid >= node_num)
            {
                return;
            }

            // Build neighbor mask (including self) and collapse topology.
            int cco_id = get_coarse_space_id(coarse_space_map, tid, level - 1);
            uint32_t neighbor = (1u << (cco_id % 32));
            int out_of_bank_neighbor_count = 0;
            for (int n = row_ptr[tid]; n < row_ptr[tid + 1]; ++n)
            {
                int neighbor_vid = cols[n];
                if (neighbor_vid == -1)
                {
                    break;
                }

                int neighbor_cco_id = get_coarse_space_id(coarse_space_map, tid, level - 1);
                if (cco_id / 32 == neighbor_cco_id / 32)
                {
                    neighbor |= (1u << (neighbor_cco_id % 32));
                }
                else
                {
                    cols[out_of_bank_neighbor_count] = neighbor_vid;
                    ++out_of_bank_neighbor_count;
                }
            }
            if (out_of_bank_neighbor_count != (row_ptr[tid + 1] - row_ptr[tid]))
            {
                cols[out_of_bank_neighbor_count] = -1;
            }

            // TODO: maybe optimize?
            // 1. write cco id to shared mem [128]
            // 2. block sort
            // 3. count block local cco id
            // 4. atomic or to block local cco neighbor slot.
            cu::atomic_ref<uint32_t> neighbor_out{neighbors[cco_id]};
            neighbor_out.fetch_or(neighbor);
        }

        /// @brief Build local CCO mapping from coarse space level x-1 -> coarse space level x.
        /// @param level Target coarse space lv x.
        /// @param cco_num Total CCO number at coarse space lv x-1.
        /// @param cco_num_per_bank Independent CCO number per bank at coarse space lv x.
        /// @param local_cco_ids Bank local CCO id at coarse space lv x.
        __global__ void build_local_cco_lvx(int level,
                                            int cco_num,
                                            ctd::span<const uint32_t> neighbors,
                                            ctd::span<const int> coarse_space_map,
                                            ctd::span<int> cco_num_per_bank,
                                            ctd::span<int> local_cco_ids)
        {

            int tid = blockDim.x * blockIdx.x + threadIdx.x; // global thread id
            int bid = tid / 32;                              // bank id
            int lid = tid % 32;                              // lane id
            if (tid >= cco_num)
            {
                return;
            }

            // Build connectivity mask (including self).
            uint32_t connection = neighbors[tid];
            uint32_t visited = (1u << lid);
            uint32_t to_visit = connection ^ visited;
            while (to_visit)
            {
                int visiting = ctd::countr_one(to_visit);
                connection |= neighbors[visiting + bid * 32];
                visited |= (1u << visiting);
                to_visit = connection ^ visited;
            }

            // Find bank (warp) local connected component (CCO).
            uint32_t cco_lead_lid = ctd::countr_one(connection);
            bool is_lead = (cco_lead_lid == lid);
            uint32_t lead_lanes = __ballot_sync(0xFFFFFFFFu, is_lead);
            uint32_t before_cco_lead_mask = static_cast<uint32_t>((1ull << cco_lead_lid) - 1ull);

            // Write cco info back.
            local_cco_ids[tid] = ctd::popcount(lead_lanes | before_cco_lead_mask);
            if (lid == 0)
            {
                int bank_cco_num = ctd::popcount(lead_lanes);
                cco_num_per_bank[bid] = bank_cco_num;
            }
        };

        /// @brief Build global coarse space level x CCO id.
        /// @param level Target coarse space lv x.
        /// @param local_cco_ids Bank local CCO id at coarse space lv x.
        /// @param cco_num_per_bank Independent CCO number per bank at coarse space lv x.
        /// @param cco_num_per_bank_summed Inclusive sum of cco_num_per_bank.
        /// @param coarse_space_map Global CCO id out.
        __global__ void build_global_cco_lvx(
            int level,
            ctd::span<const int> local_cco_ids,
            ctd::span<const int> cco_num_per_bank,
            ctd::span<const int> cco_num_per_bank_summed,
            ctd::span<int> coarse_space_map)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x; // global thread id
            int bid = tid / 32;                              // bank id

            if (tid >= local_cco_ids.size())
            {
                return;
            }

            int cco_id = local_cco_ids[tid] + cco_num_per_bank_summed[bid] - cco_num_per_bank[bid];
            coarse_space_map[tid * MAX_COARSE_LEVEL + level] = cco_id;
        }

        cu::device_buffer<int> build_coarse_space_map(ctd::span<const int> row_ptr, ctd::span<int> cols, CudaRuntime rt)
        {
            int node_num = row_ptr.size() - 1;
            int max_bank_per_level = div_round_upper(node_num, 32);
            auto local_cco_ids =
                cu::make_buffer<int>(rt.stream, rt.mr, node_num, cu::no_init);
            auto cco_num_per_bank =
                cu::make_buffer<int>(rt.stream, rt.mr, max_bank_per_level, cu::no_init);
            auto cco_num_per_bank_summed =
                cu::make_buffer<int>(rt.stream, rt.mr, max_bank_per_level, cu::no_init);
            auto coarse_space_map =
                cu::make_buffer<int>(rt.stream, rt.mr, node_num * MAX_COARSE_LEVEL, cu::no_init);

            // Build coarse map level 0.
            int grid_num = div_round_upper(node_num, 128);
            build_local_cco_lv0<<<grid_num, 128, 0, rt.stream.get()>>>(
                row_ptr, cols, cco_num_per_bank, local_cco_ids);

            size_t cub_tmp_size;
            cub::DeviceScan::InclusiveSum(nullptr,
                                          cub_tmp_size,
                                          cco_num_per_bank.data(),
                                          cco_num_per_bank_summed.data(),
                                          max_bank_per_level,
                                          rt.stream.get());
            auto cub_tmp =
                cu::make_buffer<char>(rt.stream, rt.mr, cub_tmp_size, cu::no_init);
            cub::DeviceScan::InclusiveSum(cub_tmp.data(),
                                          cub_tmp_size,
                                          cco_num_per_bank.data(),
                                          cco_num_per_bank_summed.data(),
                                          max_bank_per_level,
                                          rt.stream.get());

            build_global_cco<<<grid_num, 128, 0, rt.stream.get()>>>(
                0, local_cco_ids, cco_num_per_bank, cco_num_per_bank_summed, coarse_space_map);

            // Build coarse map level 1 ... MAX_COARSE_LEVEL-1 recursively.
            int cco_num = device2host(cco_num_per_bank_summed.data() + max_bank_per_level - 1, rt);
            auto neighbors = cu::make_buffer<uint32_t>(rt.stream, rt.mr, cco_num, cu::no_init);
            for (int lv = 1; lv < MAX_COARSE_LEVEL; ++lv)
            {
                int grid_num = div_round_upper(node_num, 128);
                build_neighbor_masks_lvx<<<grid_num, 128, 0, rt.stream.get()>>>(
                    lv, coarse_space_map, row_ptr, cols, neighbors);

                grid_num = div_round_upper(cco_num, 128);
                build_local_cco_lvx<<<grid_num, 128, 0, rt.stream.get()>>>(
                    lv, cco_num, neighbors, coarse_space_map, cco_num_per_bank, local_cco_ids);

                int bank_num = div_round_upper(cco_num, 32);
                cub::DeviceScan::InclusiveSum(cub_tmp.data(),
                                              cub_tmp_size,
                                              cco_num_per_bank.data(),
                                              cco_num_per_bank_summed.data(),
                                              bank_num,
                                              rt.stream.get());
                build_global_cco_lvx<<<grid_num, 128, 0, rt.stream.get()>>>(
                    lv, local_cco_ids, cco_num_per_bank, cco_num_per_bank_summed, coarse_space_map);

                cco_num = device2host(cco_num_per_bank_summed.data() + bank_num - 1, rt);
            }

            // This is redundant. It's placed here because of readability.
            rt.stream.sync();
            return coarse_space_map;
        }

        //     template <int D>
        //     __global__ void build_multilevel_r_kernel(
        //         const double *fine_r,
        //         double *multi_r,
        //         const int *fine_ancestors,
        //         const int *level_offsets,
        //         int max_level_count,
        //         int level_count,
        //         int fine_node_count)
        //     {
        //         int fine = blockIdx.x * blockDim.x + threadIdx.x;
        //         if (fine >= fine_node_count)
        //         {
        //             return;
        //         }
        //
        //         int ancestor_offset = fine * max_level_count;
        //         for (int d = 0; d < D; ++d)
        //         {
        //             double val = fine_r[fine * D + d];
        //             for (int level = 0; level < level_count; ++level)
        //             {
        //                 int node = fine_ancestors[ancestor_offset + level];
        //                 cu::atomic_ref<double> out{multi_r[(level_offsets[level] + node) * D + d]};
        //                 out.fetch_add(val, ctd::memory_order_relaxed);
        //             }
        //         }
        //     }
        //
        //     template <int D>
        //     __global__ void apply_bank_inverse_kernel(
        //         const double *inverse_matrices,
        //         const double *multi_r,
        //         double *multi_z,
        //         int node_offset,
        //         int matrix_offset,
        //         int bank_count)
        //     {
        //         constexpr int N = BANK_SIZE * D;
        //
        //         int bank = blockIdx.x;
        //         int row = threadIdx.x;
        //         if (bank >= bank_count || row >= N)
        //         {
        //             return;
        //         }
        //
        //         int matrix_base = (matrix_offset + bank) * N * N;
        //         int vector_base = (node_offset + bank * BANK_SIZE) * D;
        //
        //         double sum = 0.0;
        //         for (int col = 0; col < N; ++col)
        //         {
        //             sum += inverse_matrices[matrix_base + row * N + col] * multi_r[vector_base + col];
        //         }
        //         multi_z[vector_base + row] = sum;
        //     }
        //
        //     template <int D>
        //     __global__ void collect_final_z_kernel(
        //         const double *multi_z,
        //         double *fine_z,
        //         const int *fine_ancestors,
        //         const int *level_offsets,
        //         int max_level_count,
        //         int level_count,
        //         int fine_node_count)
        //     {
        //         int fine = blockIdx.x * blockDim.x + threadIdx.x;
        //         if (fine >= fine_node_count)
        //         {
        //             return;
        //         }
        //
        //         int ancestor_offset = fine * max_level_count;
        //         for (int d = 0; d < D; ++d)
        //         {
        //             double sum = 0.0;
        //             for (int level = 0; level < level_count; ++level)
        //             {
        //                 int node = fine_ancestors[ancestor_offset + level];
        //                 sum += multi_z[(level_offsets[level] + node) * D + d];
        //             }
        //             fine_z[fine * D + d] = sum;
        //         }
        //     }
        //
        //     __global__ void store_level0_mapping_kernel(
        //         const int *level0_map, int *fine_ancestors, int max_level_count, int fine_node_count)
        //     {
        //         int fine = blockIdx.x * blockDim.x + threadIdx.x;
        //         if (fine >= fine_node_count)
        //         {
        //             return;
        //         }
        //
        //         fine_ancestors[fine * max_level_count] = level0_map[fine];
        //     }
        //
        //     __global__ void build_local_connectivity_kernel(
        //         TopologyView topology,
        //         const int *fine_ancestors,
        //         int max_level_count,
        //         int level,
        //         uint32_t *connect_masks)
        //     {
        //         int fine = blockIdx.x * blockDim.x + threadIdx.x;
        //         if (fine >= topology.dim)
        //         {
        //             return;
        //         }
        //
        //         int row_node = fine_ancestors[fine * max_level_count + level];
        //         if (row_node < 0)
        //         {
        //             return;
        //         }
        //
        //         int row_bank = row_node / BANK_SIZE;
        //         for (int idx = topology.row_ptr[fine]; idx < topology.row_ptr[fine + 1]; ++idx)
        //         {
        //             int neighbor = topology.cols[idx];
        //             if (neighbor == fine)
        //             {
        //                 continue;
        //             }
        //
        //             int col_node = fine_ancestors[neighbor * max_level_count + level];
        //             if (col_node < 0 || col_node == row_node || (col_node / BANK_SIZE) != row_bank)
        //             {
        //                 continue;
        //             }
        //
        //             uint32_t bit = uint32_t{1} << (col_node % BANK_SIZE);
        //             cuda::atomic_ref<uint32_t, cuda::thread_scope_device> out(connect_masks[row_node]);
        //             out.fetch_or(bit, ctd::memory_order_relaxed);
        //         }
        //     }
        //
        //     __global__ void close_components_and_count_kernel(
        //         uint32_t *connect_masks,
        //         int current_node_count,
        //         const int *level_remap,
        //         int *component_counts)
        //     {
        //         int bank = blockIdx.x;
        //         int lane = threadIdx.x;
        //         int node = bank * BANK_SIZE + lane;
        //
        //         __shared__ uint32_t direct[BANK_SIZE];
        //         using WarpReduce = cub::WarpReduce<int, BANK_SIZE>;
        //         __shared__ typename WarpReduce::TempStorage reduce_storage;
        //
        //         bool active = node < current_node_count;
        //         if (level_remap != nullptr)
        //         {
        //             active = level_remap[node] >= 0;
        //         }
        //
        //         if (active)
        //         {
        //             direct[lane] = connect_masks[node] | (uint32_t{1} << lane);
        //         }
        //         else
        //         {
        //             direct[lane] = 0;
        //         }
        //         __syncthreads();
        //
        //         uint32_t component = direct[lane];
        //         if (active)
        //         {
        //             while (true)
        //             {
        //                 uint32_t todo = component;
        //                 bool changed = false;
        //                 while (todo)
        //                 {
        //                     int next = cuda::std::countr_zero(todo);
        //                     uint32_t expanded = component | direct[next];
        //                     changed = changed || (expanded != component);
        //                     component = expanded;
        //                     todo &= (todo - 1);
        //                 }
        //                 if (!changed)
        //                 {
        //                     break;
        //                 }
        //             }
        //             connect_masks[node] = component;
        //         }
        //
        //         uint32_t lower = (lane == 0) ? 0 : ((uint32_t{1} << lane) - 1);
        //         int representative = (active && component != 0 && cuda::std::popcount(component & lower) == 0)
        //                                  ? 1
        //                                  : 0;
        //
        //         int total = WarpReduce(reduce_storage).Sum(representative);
        //         if (lane == 0)
        //         {
        //             component_counts[bank] = total;
        //         }
        //     }
        //
        //     __global__ void compute_next_count_kernel(
        //         const int *bank_offsets, const int *component_counts, int bank_count, int *next_count)
        //     {
        //         if (threadIdx.x != 0 || blockIdx.x != 0)
        //         {
        //             return;
        //         }
        //
        //         if (bank_count == 0)
        //         {
        //             next_count[0] = 0;
        //             return;
        //         }
        //
        //         next_count[0] = bank_offsets[bank_count - 1] + component_counts[bank_count - 1];
        //     }
        //
        //     __global__ void assign_next_level_kernel(
        //         const uint32_t *connect_masks,
        //         const int *bank_offsets,
        //         int current_node_count,
        //         const int *level_remap,
        //         int *current_to_next)
        //     {
        //         int bank = blockIdx.x;
        //         int lane = threadIdx.x;
        //         int node = bank * BANK_SIZE + lane;
        //
        //         __shared__ int representative_prefix[BANK_SIZE];
        //         using WarpScan = cub::WarpScan<int, BANK_SIZE>;
        //         __shared__ typename WarpScan::TempStorage scan_storage;
        //         bool active = node < current_node_count;
        //         if (level_remap != nullptr)
        //         {
        //             active = level_remap[node] >= 0;
        //         }
        //         uint32_t component = active ? connect_masks[node] : 0;
        //         uint32_t lower = (lane == 0) ? 0 : ((uint32_t{1} << lane) - 1);
        //         int representative =
        //             (active && cuda::std::popcount(component & lower) == 0) ? 1 : 0;
        //
        //         int prefix = 0;
        //         WarpScan(scan_storage).ExclusiveSum(representative, prefix);
        //         if (representative)
        //         {
        //             representative_prefix[lane] = prefix;
        //         }
        //         __syncthreads();
        //
        //         if (active)
        //         {
        //             int leader = cuda::std::countr_zero(component);
        //             current_to_next[node] = bank_offsets[bank] + representative_prefix[leader];
        //         }
        //     }
        //
        //     __global__ void compose_fine_ancestors_kernel(
        //         int *fine_ancestors,
        //         int max_level_count,
        //         int level,
        //         const int *current_to_next,
        //         int fine_node_count)
        //     {
        //         int fine = blockIdx.x * blockDim.x + threadIdx.x;
        //         if (fine >= fine_node_count)
        //         {
        //             return;
        //         }
        //
        //         int current = fine_ancestors[fine * max_level_count + level];
        //         fine_ancestors[fine * max_level_count + level + 1] = current_to_next[current];
        //     }
        //
        //     template <int D>
        //     __global__ void init_level0_ghost_identity_kernel(
        //         double *local_matrices, const int *level0_remap, int matrix_offset, int bank_count)
        //     {
        //         int node = blockIdx.x * blockDim.x + threadIdx.x;
        //         if (node >= bank_count * BANK_SIZE)
        //         {
        //             return;
        //         }
        //
        //         if (level0_remap[node] >= 0)
        //         {
        //             return;
        //         }
        //
        //         constexpr int N = BANK_SIZE * D;
        //         int bank = node / BANK_SIZE;
        //         int local = node % BANK_SIZE;
        //         int base = (matrix_offset + bank) * N * N;
        //         for (int d = 0; d < D; ++d)
        //         {
        //             int row = local * D + d;
        //             local_matrices[base + row * N + row] = 1.0;
        //         }
        //     }
        //
        //     template <int D>
        //     __global__ void init_padded_identity_kernel(
        //         double *local_matrices, int node_count, int matrix_offset, int bank_count)
        //     {
        //         int node = blockIdx.x * blockDim.x + threadIdx.x;
        //         if (node >= bank_count * BANK_SIZE || node < node_count)
        //         {
        //             return;
        //         }
        //
        //         constexpr int N = BANK_SIZE * D;
        //         int bank = node / BANK_SIZE;
        //         int local = node % BANK_SIZE;
        //         int base = (matrix_offset + bank) * N * N;
        //         for (int d = 0; d < D; ++d)
        //         {
        //             int row = local * D + d;
        //             local_matrices[base + row * N + row] = 1.0;
        //         }
        //     }
        //
        //     template <int D>
        //     __global__ void assemble_level_matrices_kernel(
        //         BCOOView matrix,
        //         const int *fine_ancestors,
        //         int max_level_count,
        //         int level,
        //         int matrix_offset,
        //         double *local_matrices)
        //     {
        //         int idx = blockIdx.x * blockDim.x + threadIdx.x;
        //         if (idx >= matrix.non_zeros)
        //         {
        //             return;
        //         }
        //
        //         int row = matrix.rows[idx];
        //         int col = matrix.cols[idx];
        //         int row_node = fine_ancestors[row * max_level_count + level];
        //         int col_node = fine_ancestors[col * max_level_count + level];
        //         if (row_node < 0 || col_node < 0 || row_node / BANK_SIZE != col_node / BANK_SIZE)
        //         {
        //             return;
        //         }
        //
        //         constexpr int N = BANK_SIZE * D;
        //         int matrix_base = (matrix_offset + row_node / BANK_SIZE) * N * N;
        //         int local_row = (row_node % BANK_SIZE) * D;
        //         int local_col = (col_node % BANK_SIZE) * D;
        //         int value_base = idx * D * D;
        //         for (int i = 0; i < D; ++i)
        //         {
        //             for (int j = 0; j < D; ++j)
        //             {
        //                 cu::atomic_ref<double> out{
        //                     local_matrices[matrix_base + (local_row + i) * N + local_col + j]};
        //                 out.fetch_add(matrix.vals[value_base + i * D + j], ctd::memory_order_relaxed);
        //             }
        //         }
        //     }
        //
        //     template <int D>
        //     __global__ void assemble_all_levels_matrices_kernel(
        //         BCOOView matrix,
        //         const int *fine_ancestors,
        //         int max_level_count,
        //         int level_count,
        //         const int *level_matrix_offsets,
        //         double *local_matrices)
        //     {
        //         int idx = blockIdx.x * blockDim.x + threadIdx.x;
        //         if (idx >= matrix.non_zeros)
        //         {
        //             return;
        //         }
        //
        //         int row = matrix.rows[idx];
        //         int col = matrix.cols[idx];
        //         constexpr int N = BANK_SIZE * D;
        //         int value_base = idx * D * D;
        //
        //         for (int level = 0; level < level_count; ++level)
        //         {
        //             int row_node = fine_ancestors[row * max_level_count + level];
        //             int col_node = fine_ancestors[col * max_level_count + level];
        //             if (row_node < 0 || col_node < 0 || row_node / BANK_SIZE != col_node / BANK_SIZE)
        //             {
        //                 continue;
        //             }
        //
        //             int matrix_base = (level_matrix_offsets[level] + row_node / BANK_SIZE) * N * N;
        //             int local_row = (row_node % BANK_SIZE) * D;
        //             int local_col = (col_node % BANK_SIZE) * D;
        //             for (int i = 0; i < D; ++i)
        //             {
        //                 for (int j = 0; j < D; ++j)
        //                 {
        //                     cu::atomic_ref<double> out{
        //                         local_matrices[matrix_base + (local_row + i) * N + local_col + j]};
        //                     out.fetch_add(matrix.vals[value_base + i * D + j], ctd::memory_order_relaxed);
        //                 }
        //             }
        //         }
        //     }
        //
        //     template <int D>
        //     __global__ void invert_banks_kernel(
        //         double *local_matrices, double *inverse_matrices, int matrix_offset, int bank_count)
        //     {
        //         constexpr int N = BANK_SIZE * D;
        //
        //         int bank = blockIdx.x;
        //         int row = threadIdx.x;
        //         if (bank >= bank_count || row >= N)
        //         {
        //             return;
        //         }
        //
        //         __shared__ int failed;
        //         __shared__ int swap_row;
        //         if (row == 0)
        //         {
        //             failed = 0;
        //             swap_row = 0;
        //         }
        //         __syncthreads();
        //
        //         int base = (matrix_offset + bank) * N * N;
        //         for (int col = 0; col < N; ++col)
        //         {
        //             inverse_matrices[base + row * N + col] = (row == col) ? 1.0 : 0.0;
        //         }
        //         __syncthreads();
        //
        //         for (int pivot = 0; pivot < N; ++pivot)
        //         {
        //             if (row == 0)
        //             {
        //                 int best_row = pivot;
        //                 double best_abs = ctd::abs(local_matrices[base + pivot * N + pivot]);
        //                 for (int candidate = pivot + 1; candidate < N; ++candidate)
        //                 {
        //                     double candidate_abs =
        //                         ctd::abs(local_matrices[base + candidate * N + pivot]);
        //                     if (candidate_abs > best_abs)
        //                     {
        //                         best_abs = candidate_abs;
        //                         best_row = candidate;
        //                     }
        //                 }
        //
        //                 if (best_abs < 1e-20)
        //                 {
        //                     failed = 1;
        //                 }
        //                 else
        //                 {
        //                     swap_row = best_row;
        //                 }
        //             }
        //             __syncthreads();
        //
        //             if (failed)
        //             {
        //                 break;
        //             }
        //
        //             if (swap_row != pivot && (row == pivot || row == swap_row))
        //             {
        //                 int other = (row == pivot) ? swap_row : pivot;
        //                 for (int col = 0; col < N; ++col)
        //                 {
        //                     double tmp_a = local_matrices[base + row * N + col];
        //                     local_matrices[base + row * N + col] =
        //                         local_matrices[base + other * N + col];
        //                     local_matrices[base + other * N + col] = tmp_a;
        //
        //                     double tmp_inv = inverse_matrices[base + row * N + col];
        //                     inverse_matrices[base + row * N + col] =
        //                         inverse_matrices[base + other * N + col];
        //                     inverse_matrices[base + other * N + col] = tmp_inv;
        //                 }
        //             }
        //             __syncthreads();
        //
        //             double pivot_value = local_matrices[base + pivot * N + pivot];
        //             if (row == pivot)
        //             {
        //                 double inv_pivot = 1.0 / pivot_value;
        //                 for (int col = 0; col < N; ++col)
        //                 {
        //                     local_matrices[base + pivot * N + col] *= inv_pivot;
        //                     inverse_matrices[base + pivot * N + col] *= inv_pivot;
        //                 }
        //             }
        //             __syncthreads();
        //
        //             if (row != pivot)
        //             {
        //                 double factor = local_matrices[base + row * N + pivot];
        //                 if (factor != 0.0)
        //                 {
        //                     for (int col = 0; col < N; ++col)
        //                     {
        //                         local_matrices[base + row * N + col] -=
        //                             factor * local_matrices[base + pivot * N + col];
        //                         inverse_matrices[base + row * N + col] -=
        //                             factor * inverse_matrices[base + pivot * N + col];
        //                     }
        //                 }
        //             }
        //             __syncthreads();
        //         }
        //
        //         if (failed)
        //         {
        //             for (int col = 0; col < N; ++col)
        //             {
        //                 inverse_matrices[base + row * N + col] = (row == col) ? 1.0 : 0.0;
        //             }
        //         }
        //     }
        //
        //     void build_fine_slot_mapping(
        //         int part_num,
        //         ctd::span<const int> part_id,
        //         std::vector<int> &fine_to_slot,
        //         std::vector<int> &slot_to_fine)
        //     {
        //     }
        // } // namespace
        //
        // void MASPreconditioner::analyze_pattern(
        //     const BCOOMatrix &A, CudaRuntime rt)
        // {
        //     // Metis partition.
        //     TopologyView h_topo = A.host_topology_view();
        //     int part_num;
        //     std::vector<int> part_id;
        //     metis_partition(h_topo.row_ptr, h_topo.cols, BANK_SIZE, part_num, part_id);
        //
        //     // Build fine slot mapping.
        //     // fine to slot: vert id -> padded reordered id.
        //     // slot to fine: padded reorder id -> vert id. -1 indicates virtual padding.
        //     std::vector<int> fine_to_slot(part_id.size(), -1);
        //     std::vector<int> slot_to_fine(part_num * BANK_SIZE, -1);
        //     std::vector<int> bank_fill_counter(part_id.size(), 0);
        //     for (int i = 0; i < part_id.size(); ++i)
        //     {
        //         int id = part_id[i];
        //         int &fill = bank_fill_counter[id];
        //         int slot = id * BANK_SIZE + fill;
        //         fine_to_slot[i] = slot;
        //         slot_to_fine[slot] = i;
        //         fill += 1;
        //     }
        //     fine_to_slot_ = cu::make_buffer<int>(rt.stream, rt.mr, fine_to_slot.size(), cu::no_init);
        //     cu::copy_bytes(rt.stream, fine_to_slot, fine_to_slot_);
        //     slot_to_fine_ = cu::make_buffer<int>(rt.stream, rt.mr, slot_to_fine.size(), cu::no_init);
        //     cu::copy_bytes(rt.stream, slot_to_fine, slot_to_fine_);
        //
        //     int fine_num = A.view().dim;
        //     fine_ancestors_ = cu::make_buffer<int>(rt.stream, rt.mr, fine_num * MAX_COARSEN_LEVEL, cu::no_init);
        //     cu::fill_bytes(rt.stream, *fine_ancestors_, ctd::uint8_t{0xff});

        // int fine_grid = div_round_upper(fine_node_count_, KERNEL_BLOCK_SIZE);
        // store_level0_mapping_kernel<<<fine_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
        //     fine_to_slot_->data(), fine_ancestors_->data(), MAX_COARSEN_LEVEL, fine_node_count_);
        //
        // connect_masks_ =
        //     cu::make_buffer<uint32_t>(rt.stream, rt.mr, level0_padded_count, cu::no_init);
        // component_counts_ = cu::make_buffer<int>(rt.stream, rt.mr, max_bank_count_, cu::no_init);
        // bank_offsets_ = cu::make_buffer<int>(rt.stream, rt.mr, max_bank_count_, cu::no_init);
        // current_to_next_ = cu::make_buffer<int>(rt.stream, rt.mr, level0_padded_count, cu::no_init);
        // next_count_ = cu::make_buffer<int>(rt.stream, rt.mr, 1, cu::no_init);
        //
        // size_t scan_bytes = 1;
        // if (max_bank_count_ > 0)
        // {
        //     cub::DeviceScan::ExclusiveSum(
        //         nullptr,
        //         scan_bytes,
        //         component_counts_->data(),
        //         bank_offsets_->data(),
        //         max_bank_count_,
        //         rt.stream.get());
        // }
        // scan_storage_ = cu::make_buffer<char>(rt.stream, rt.mr, scan_bytes, cu::no_init);
        //
        // levels_.clear();
        // level_node_offsets_.clear();
        // level_matrix_offsets_.clear();
        // levels_.push_back(LevelInfo{level0_padded_count, level0_padded_count, lv0_bank_num});
        //
        // int current_node_count = level0_padded_count;
        // int current_bank_count = lv0_bank_num;
        // int current_level = 0;
        //     while (current_level + 1 < MAX_COARSEN_LEVEL && current_node_count > 1)
        //     {
        //         cudaMemsetAsync(
        //             connect_masks_->data(),
        //             0,
        //             current_bank_count * BANK_SIZE * sizeof(uint32_t),
        //             rt.stream.get());
        //
        //         build_local_connectivity_kernel<<<fine_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
        //             topology,
        //             fine_ancestors_->data(),
        //             MAX_COARSEN_LEVEL,
        //             current_level,
        //             connect_masks_->data());
        //
        //         close_components_and_count_kernel<<<current_bank_count, BANK_SIZE, 0, rt.stream.get()>>>(
        //             connect_masks_->data(),
        //             current_node_count,
        //             current_level == 0 ? slot_to_fine_->data() : nullptr,
        //             component_counts_->data());
        //
        //         size_t scan_bytes = scan_storage_->size();
        //         cub::DeviceScan::ExclusiveSum(
        //             scan_storage_->data(),
        //             scan_bytes,
        //             component_counts_->data(),
        //             bank_offsets_->data(),
        //             current_bank_count,
        //             rt.stream.get());
        //         compute_next_count_kernel<<<1, 1, 0, rt.stream.get()>>>(
        //             bank_offsets_->data(), component_counts_->data(), current_bank_count, next_count_->data());
        //
        //         int next_node_count = device2host(next_count_->data(), rt);
        //         if (next_node_count >= current_node_count)
        //         {
        //             break;
        //         }
        //
        //         assign_next_level_kernel<<<current_bank_count, BANK_SIZE, 0, rt.stream.get()>>>(
        //             connect_masks_->data(),
        //             bank_offsets_->data(),
        //             current_node_count,
        //             current_level == 0 ? slot_to_fine_->data() : nullptr,
        //             current_to_next_->data());
        //         compose_fine_ancestors_kernel<<<fine_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
        //             fine_ancestors_->data(),
        //             MAX_COARSEN_LEVEL,
        //             current_level,
        //             current_to_next_->data(),
        //             fine_node_count_);
        //
        //         int next_bank_count = (next_node_count + BANK_SIZE - 1) / BANK_SIZE;
        //         levels_.push_back(LevelInfo{next_node_count, next_bank_count * BANK_SIZE, next_bank_count});
        //         current_node_count = next_node_count;
        //         current_bank_count = next_bank_count;
        //         current_level++;
        //     }
        //
        //     level_count_ = levels_.size();
        //     total_padded_nodes_ = 0;
        //     int total_bank_count = 0;
        //     for (auto &level : levels_)
        //     {
        //         level_node_offsets_.push_back(total_padded_nodes_);
        //         total_padded_nodes_ += level.padded_count;
        //         level_matrix_offsets_.push_back(total_bank_count);
        //         total_bank_count += level.bank_count;
        //     }
        //
        //     level_node_offsets_device_ =
        //         cu::make_buffer<int>(rt.stream, rt.mr, level_node_offsets_.size(), cu::no_init);
        //     cu::copy_bytes(rt.stream, level_node_offsets_, *level_node_offsets_device_);
        //     level_matrix_offsets_device_ =
        //         cu::make_buffer<int>(rt.stream, rt.mr, level_matrix_offsets_.size(), cu::no_init);
        //     cu::copy_bytes(rt.stream, level_matrix_offsets_, *level_matrix_offsets_device_);
        //
        //     int total_matrix_size = 0;
        //     for (auto &level : levels_)
        //     {
        //         total_matrix_size += level.bank_count;
        //     }
        //     int local_dim = BANK_SIZE * block_dim_;
        //     int matrix_entries = total_matrix_size * local_dim * local_dim;
        //     local_matrices_ = cu::make_buffer<double>(rt.stream, rt.mr, matrix_entries, cu::no_init);
        //     inverse_matrices_ = cu::make_buffer<double>(rt.stream, rt.mr, matrix_entries, cu::no_init);
        //     multi_level_r_ =
        //         cu::make_buffer<double>(rt.stream, rt.mr, total_padded_nodes_ *block_dim_, cu::no_init);
        //     multi_level_z_ =
        //         cu::make_buffer<double>(rt.stream, rt.mr, total_padded_nodes_ *block_dim_, cu::no_init);
        //     rt.stream.sync();
        // } // namespace
        //
        // void MASPreconditioner::factorize(const BCOOMatrix &A, CudaRuntime rt)
        // {
        //     BCOOView matrix = A.view();
        //     int old_block_dim = block_dim_;
        //     block_dim_ = matrix.block_dim;
        //
        //     TopologyView h_topo = A.host_topology_view();
        //     TopologyView d_topo = A.device_topology_view();
        //
        //     cudaMemsetAsync(
        //         local_matrices_->data(),
        //         0,
        //         local_matrices_->size() * sizeof(double),
        //         rt.stream.get());
        //
        //     int matrix_grid = div_round_upper(matrix.non_zeros, KERNEL_BLOCK_SIZE);
        //     for (int current_level = 0; current_level < level_count_; ++current_level)
        //     {
        //         auto &level = levels_[current_level];
        //         int init_grid = div_round_upper(level.padded_count, KERNEL_BLOCK_SIZE);
        //         switch (block_dim_)
        //         {
        //         case 1:
        //             if (current_level == 0)
        //             {
        //                 init_level0_ghost_identity_kernel<1>
        //                     <<<init_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
        //                         local_matrices_->data(), slot_to_fine_->data(), level_matrix_offsets_[0],
        //                         level.bank_count);
        //             }
        //             else
        //             {
        //                 init_padded_identity_kernel<1><<<init_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
        //                     local_matrices_->data(),
        //                     level.node_count,
        //                     level_matrix_offsets_[current_level],
        //                     level.bank_count);
        //             }
        //             break;
        //         case 2:
        //             if (current_level == 0)
        //             {
        //                 init_level0_ghost_identity_kernel<2>
        //                     <<<init_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
        //                         local_matrices_->data(), slot_to_fine_->data(), level_matrix_offsets_[0],
        //                         level.bank_count);
        //             }
        //             else
        //             {
        //                 init_padded_identity_kernel<2><<<init_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
        //                     local_matrices_->data(),
        //                     level.node_count,
        //                     level_matrix_offsets_[current_level],
        //                     level.bank_count);
        //             }
        //             break;
        //         case 3:
        //             if (current_level == 0)
        //             {
        //                 init_level0_ghost_identity_kernel<3>
        //                     <<<init_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
        //                         local_matrices_->data(), slot_to_fine_->data(), level_matrix_offsets_[0],
        //                         level.bank_count);
        //             }
        //             else
        //             {
        //                 init_padded_identity_kernel<3><<<init_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
        //                     local_matrices_->data(),
        //                     level.node_count,
        //                     level_matrix_offsets_[current_level],
        //                     level.bank_count);
        //             }
        //             break;
        //         default:
        //             throw std::runtime_error("[CudaPCG] Unsupported block size.");
        //         }
        //     }
        //
        //     switch (block_dim_)
        //     {
        //     case 1:
        //         assemble_all_levels_matrices_kernel<1><<<matrix_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
        //             matrix,
        //             fine_ancestors_->data(),
        //             MAX_COARSEN_LEVEL,
        //             level_count_,
        //             level_matrix_offsets_device_->data(),
        //             local_matrices_->data());
        //         break;
        //     case 2:
        //         assemble_all_levels_matrices_kernel<2><<<matrix_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
        //             matrix,
        //             fine_ancestors_->data(),
        //             MAX_COARSEN_LEVEL,
        //             level_count_,
        //             level_matrix_offsets_device_->data(),
        //             local_matrices_->data());
        //         break;
        //     case 3:
        //         assemble_all_levels_matrices_kernel<3><<<matrix_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
        //             matrix,
        //             fine_ancestors_->data(),
        //             MAX_COARSEN_LEVEL,
        //             level_count_,
        //             level_matrix_offsets_device_->data(),
        //             local_matrices_->data());
        //         break;
        //     default:
        //         throw std::runtime_error("[CudaPCG] Unsupported block size.");
        //     }
        //
        //     for (int current_level = 0; current_level < level_count_; ++current_level)
        //     {
        //         auto &level = levels_[current_level];
        //         switch (block_dim_)
        //         {
        //         case 1:
        //             invert_banks_kernel<1><<<level.bank_count, BANK_SIZE, 0, rt.stream.get()>>>(
        //                 local_matrices_->data(),
        //                 inverse_matrices_->data(),
        //                 level_matrix_offsets_[current_level],
        //                 level.bank_count);
        //             break;
        //         case 2:
        //             invert_banks_kernel<2><<<level.bank_count, BANK_SIZE * 2, 0, rt.stream.get()>>>(
        //                 local_matrices_->data(),
        //                 inverse_matrices_->data(),
        //                 level_matrix_offsets_[current_level],
        //                 level.bank_count);
        //             break;
        //         case 3:
        //             invert_banks_kernel<3><<<level.bank_count, BANK_SIZE * 3, 0, rt.stream.get()>>>(
        //                 local_matrices_->data(),
        //                 inverse_matrices_->data(),
        //                 level_matrix_offsets_[current_level],
        //                 level.bank_count);
        //             break;
        //         default:
        //             throw std::runtime_error("[CudaPCG] Unsupported block size.");
        //         }
        //     }
        //
        //     rt.stream.sync();
        // }
        //
        // void MASPreconditioner::apply(ctd::span<const double> r, ctd::span<double> z, CudaRuntime rt)
        // {
        //     if (empty())
        //     {
        //         throw std::runtime_error("[CudaPCG] MAS preconditioner is not initialized.");
        //     }
        //
        //     cudaMemsetAsync(
        //         multi_level_r_->data(), 0, multi_level_r_->size() * sizeof(double), rt.stream.get());
        //
        //     int fine_grid = div_round_upper(fine_node_count_, KERNEL_BLOCK_SIZE);
        //     switch (block_dim_)
        //     {
        //     case 1:
        //         build_multilevel_r_kernel<1><<<fine_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
        //             r.data(),
        //             multi_level_r_->data(),
        //             fine_ancestors_->data(),
        //             level_node_offsets_device_->data(),
        //             MAX_COARSEN_LEVEL,
        //             level_count_,
        //             fine_node_count_);
        //         break;
        //     case 2:
        //         build_multilevel_r_kernel<2><<<fine_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
        //             r.data(),
        //             multi_level_r_->data(),
        //             fine_ancestors_->data(),
        //             level_node_offsets_device_->data(),
        //             MAX_COARSEN_LEVEL,
        //             level_count_,
        //             fine_node_count_);
        //         break;
        //     case 3:
        //         build_multilevel_r_kernel<3><<<fine_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
        //             r.data(),
        //             multi_level_r_->data(),
        //             fine_ancestors_->data(),
        //             level_node_offsets_device_->data(),
        //             MAX_COARSEN_LEVEL,
        //             level_count_,
        //             fine_node_count_);
        //         break;
        //     default:
        //         throw std::runtime_error("[CudaPCG] Unsupported block size.");
        //     }
        //
        //     int current_level = 0;
        //     for (auto &level : levels_)
        //     {
        //         switch (block_dim_)
        //         {
        //         case 1:
        //             apply_bank_inverse_kernel<1><<<level.bank_count, BANK_SIZE, 0, rt.stream.get()>>>(
        //                 inverse_matrices_->data(),
        //                 multi_level_r_->data(),
        //                 multi_level_z_->data(),
        //                 level_node_offsets_[current_level],
        //                 level_matrix_offsets_[current_level],
        //                 level.bank_count);
        //             break;
        //         case 2:
        //             apply_bank_inverse_kernel<2><<<level.bank_count, BANK_SIZE * 2, 0, rt.stream.get()>>>(
        //                 inverse_matrices_->data(),
        //                 multi_level_r_->data(),
        //                 multi_level_z_->data(),
        //                 level_node_offsets_[current_level],
        //                 level_matrix_offsets_[current_level],
        //                 level.bank_count);
        //             break;
        //         case 3:
        //             apply_bank_inverse_kernel<3><<<level.bank_count, BANK_SIZE * 3, 0, rt.stream.get()>>>(
        //                 inverse_matrices_->data(),
        //                 multi_level_r_->data(),
        //                 multi_level_z_->data(),
        //                 level_node_offsets_[current_level],
        //                 level_matrix_offsets_[current_level],
        //                 level.bank_count);
        //             break;
        //         default:
        //             throw std::runtime_error("[CudaPCG] Unsupported block size.");
        //         }
        //         current_level++;
        //     }
        //
        //     switch (block_dim_)
        //     {
        //     case 1:
        //         collect_final_z_kernel<1><<<fine_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
        //             multi_level_z_->data(),
        //             z.data(),
        //             fine_ancestors_->data(),
        //             level_node_offsets_device_->data(),
        //             MAX_COARSEN_LEVEL,
        //             level_count_,
        //             fine_node_count_);
        //         break;
        //     case 2:
        //         collect_final_z_kernel<2><<<fine_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
        //             multi_level_z_->data(),
        //             z.data(),
        //             fine_ancestors_->data(),
        //             level_node_offsets_device_->data(),
        //             MAX_COARSEN_LEVEL,
        //             level_count_,
        //             fine_node_count_);
        //         break;
        //     case 3:
        //         collect_final_z_kernel<3><<<fine_grid, KERNEL_BLOCK_SIZE, 0, rt.stream.get()>>>(
        //             multi_level_z_->data(),
        //             z.data(),
        //             fine_ancestors_->data(),
        //             level_node_offsets_device_->data(),
        //             MAX_COARSEN_LEVEL,
        //             level_count_,
        //             fine_node_count_);
        //         break;
        //     default:
        //         throw std::runtime_error("[CudaPCG] Unsupported block size.");
        //     }
        // }

    } // namespace polysolve::linear::mas
