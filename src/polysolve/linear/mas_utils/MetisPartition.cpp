#include <polysolve/linear/mas_utils/MetisPartition.hpp>

#include <polysolve/linear/mas_utils/CudaUtils.cuh>

#include <cuda/std/span>
#include <vector>

#define IDXTYPEWIDTH 32
#define REALTYPEWIDTH 32
#include <metis.h>
#undef IDXTYPEWIDTH
#undef REALTYPEWIDTH

namespace polysolve::linear::mas
{
    void metis_partition(ctd::span<const int> row_ptr,
                         ctd::span<const int> cols,
                         int max_part_size,
                         int &part_num,
                         std::vector<int> &part_id)
    {
        int node_num = row_ptr.size() - 1;
        assert(node_num >= 0);

        part_id.resize(node_num, 0);
        if (node_num <= max_part_size)
        {
            part_num = 1;
            return;
        }

        // idx_t is int32_t. For all major platforms int == int32_t.
        static_assert(sizeof(int) == 4, "int is not 32 bits");
        const idx_t *xadj = reinterpret_cast<const idx_t *>(row_ptr.data());
        const idx_t *adjncy = reinterpret_cast<const idx_t *>(cols.data());

        for (int slack = 0; slack < max_part_size; ++slack)
        {
            // Eq 7. from stiffgipc paper.
            int ideal_part_size = max_part_size - slack;
            part_num = (node_num + ideal_part_size - 1) / ideal_part_size;
            assert(part_num >= 1);

            idx_t ncon = 1;
            idx_t objval = 0;
            part_id.resize(part_num, 0);
            int ret = METIS_PartGraphKway(
                reinterpret_cast<idx_t *>(node_num),
                &ncon,                       // balancing constraint num (none)
                const_cast<idx_t *>(xadj),   // adjacency in CSR format
                const_cast<idx_t *>(adjncy), // adjacency in CSR format
                nullptr,                     // vertex weight (none)
                nullptr,                     // for communication volume (not used)
                nullptr,                     // edge weight (none)
                reinterpret_cast<idx_t *>(part_num),
                nullptr,                                  // target partition weight (none)
                nullptr,                                  // load imbalance tol (none)
                nullptr,                                  // options (none)
                &objval,                                  // communication volume out (not used)
                reinterpret_cast<idx_t *>(part_id.data()) // partition index out
            );
            if (ret != METIS_OK)
            {
                throw std::runtime_error("[CudaPCG] METIS partition failed.");
            }

            std::vector<int> part_size(part_num, 0);
            int current_max = 0;
            for (int i = 0; i < node_num; ++i)
            {
                int p = part_id[i];
                assert(p >= 0 && p < part_num);
                part_size[p] += 1;
                current_max = std::max(current_max, part_size[p]);
            }

            // We require all metis partition size be smaller than max_part_size
            // Else we re-partition.
            if (current_max <= max_part_size)
            {
                return;
            }
        }

        throw std::runtime_error("[CudaPCG] METIS partition size constraint failed.");
    }

} // namespace polysolve::linear::mas
