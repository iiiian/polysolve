if(TARGET mschol::mschol)
    return()
endif()

message(STATUS "Third-party: creating target 'mschol::mschol'")

include(CPM)
CPMAddPackage(
    NAME mschol
    GIT_REPOSITORY https://github.com/maxpaik16/mschol
    GIT_TAG 0ce0550f1afcedc20e46e7f9c0fb15ad67eb97c1
)

#add_library(mschol::mschol ALIAS mschol)
#target_link_libraries(mschol PRIVATE Eigen3::Eigen)
target_include_directories(mschol SYSTEM PUBLIC ${mschol_SOURCE_DIR}/src)