set(SOURCES
    FEMSolver.cpp
    FEMSolver.hpp
    Solver.cpp
    Solver.hpp
    AMGCL.cpp
    AMGCL.hpp
    EigenSolver.hpp
    EigenSolver.tpp
    HypreSolver.cpp
    HypreSolver.hpp
    Pardiso.cpp
    Pardiso.hpp
    SaddlePointSolver.cpp
    SaddlePointSolver.hpp
)

if(POLYSOLVE_WITH_CUSOLVER)
    list(APPEND SOURCES
        CuSolverDN.cu
        CuSolverDN.cuh
    )
endif()

if(POLYSOLVE_WITH_CUDSS)
    list(APPEND SOURCES
        CuDSS.cpp
        CuDSS.hpp
    )
endif()

if(POLYSOLVE_WITH_CUDA_PCG)
    list(APPEND SOURCES
        PCGSolver.cu
        PCGSolver.hpp
    )
    target_compile_options(polysolve_linear PRIVATE
        # enable __device__ lambda
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        # prevent error from Eigen header
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endif()

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polysolve_linear PRIVATE ${SOURCES})
