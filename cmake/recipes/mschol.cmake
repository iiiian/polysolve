if(TARGET mschol::mschol)
    return()
endif()

message(STATUS "Third-party: creating target 'mschol::mschol'")

include(CPM)
CPMAddPackage(
    NAME mschol
    GIT_REPOSITORY https://github.com/maxpaik16/mschol
    GIT_TAG 98481a9b270fb8e6291b82ee498e28ead31923fd
)

target_include_directories(mschol SYSTEM PUBLIC ${mschol_SOURCE_DIR}/src)