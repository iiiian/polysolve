set(SOURCES
    FEMSolver.cpp
    FEMSolver.hpp
    Solver.cpp
    Solver.hpp
    AMGCL.cpp
    AMGCL.hpp
    CuSolverDN.cu
    CuSolverDN.cuh
    EigenSolver.hpp
    EigenSolver.tpp
    HypreSolver.cpp
    HypreSolver.hpp
    Pardiso.cpp
    Pardiso.hpp
    SaddlePointSolver.cpp
    SaddlePointSolver.hpp
)

if(POLYSOLVE_WITH_CUDA)
    list(APPEND SOURCES
        PCGSolver.cu
        PCGSolver.hpp
        mas_utils/BCOOMatrix.cpp
        mas_utils/BCOOMatrix.hpp
        mas_utils/CudaUtils.cuh
        mas_utils/InnerProduct.cu
        mas_utils/InnerProduct.hpp
        mas_utils/Inverse.cu
        mas_utils/Inverse.cuh
        mas_utils/SimpleLinalg.cuh
        mas_utils/Spmv.cu
        mas_utils/Spmv.hpp
    )
endif()

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polysolve_linear PRIVATE ${SOURCES})
