if(TARGET mschol::mschol)
    return()
endif()

message(STATUS "Third-party: creating target 'mschol::mschol'")

include(CPM)
CPMAddPackage(
    NAME mschol
    GIT_REPOSITORY https://github.com/maxpaik16/mschol
    GIT_TAG d0429df4ab6a50df4b4aa8bcf00de433718470df
)

#add_library(mschol::mschol ALIAS mschol)
target_include_directories(mschol SYSTEM PUBLIC ${mschol_SOURCE_DIR}/src)