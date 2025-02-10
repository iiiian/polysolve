if(TARGET mschol::mschol)
    return()
endif()

message(STATUS "Third-party: creating target 'mschol::mschol'")

include(CPM)
CPMAddPackage(
    NAME mschol
    GIT_REPOSITORY https://github.com/maxpaik16/mschol
    GIT_TAG 8d8aa1a5fa219f5683ce81d6c1f0d84ba479f0e5
)

#add_library(mschol::mschol ALIAS mschol)
target_link_libraries(mschol PRIVATE Eigen3::Eigen)
target_include_directories(mschol SYSTEM PUBLIC ${mschol_SOURCE_DIR}/src)