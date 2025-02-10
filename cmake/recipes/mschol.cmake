if(TARGET mschol::mschol)
    return()
endif()

message(STATUS "Third-party: creating target 'mschol::mschol'")

include(CPM)
CPMAddPackage(
    NAME mschol
    GIT_REPOSITORY https://github.com/maxpaik16/mschol
    GIT_TAG 3e8a7bc84837c12261b00a1db3764e422656e4ca
)

#add_library(mschol::mschol ALIAS mschol)
#target_link_libraries(mschol PRIVATE Eigen3::Eigen)
target_include_directories(mschol SYSTEM PUBLIC ${mschol_SOURCE_DIR}/src)