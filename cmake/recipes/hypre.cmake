# HYPRE GNU Lesser General Public License

if(TARGET HYPRE::HYPRE)
    return()
endif()

message(STATUS "Third-party: creating target 'HYPRE::HYPRE'")

set(HYPRE_ENABLE_MPI    OFF CACHE INTERNAL "" FORCE)
set(HYPRE_ENABLE_PRINT_ERRORS  ON CACHE INTERNAL "" FORCE)
set(HYPRE_ENABL_BIGINT        ON CACHE INTERNAL "" FORCE)
set(HYPRE_ENABLE_FEI    OFF CACHE INTERNAL "" FORCE)
set(HYPRE_ENABLE_OPENMP ON CACHE INTERNAL "" FORCE)
set(HYPRE_SHARED       OFF CACHE INTERNAL "" FORCE)

include(CPM)
CPMAddPackage(
    NAME hypre
    GITHUB_REPOSITORY hypre-space/hypre
    GIT_TAG v2.33.0
    SOURCE_SUBDIR src
)
file(REMOVE "${hypre_SOURCE_DIR}/src/utilities/version")
