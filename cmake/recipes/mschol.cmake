if(TARGET mschol::mschol)
    return()
endif()

message(STATUS "Third-party: creating target 'mschol::mschol'")

include(CPM)
CPMAddPackage(
    NAME mschol
    GIT_REPOSITORY https://github.com/maxpaik16/mschol
    GIT_TAG 30ed45cf685379d0cdb063549bf2b4f5a8916646
)

#add_library(mschol::mschol ALIAS mschol)
target_include_directories(mschol SYSTEM PUBLIC ${mschol_SOURCE_DIR}/src)