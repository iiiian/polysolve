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

if(POLYSOLVE_WITH_CUDA_PCG)
    list(APPEND SOURCES
        PCGSolver.cu
        PCGSolver.hpp
    )
endif()

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polysolve_linear PRIVATE ${SOURCES})
