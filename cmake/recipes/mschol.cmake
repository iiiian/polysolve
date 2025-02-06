if(TARGET mschol::mschol)
    return()
endif()

message(STATUS "Third-party: creating target 'mschol::mschol'")

include(CPM)
CPMAddPackage(
    NAME mschol
    GIT_REPOSITORY https://github.com/maxpaik16/mschol
    GIT_TAG 3a4409a53877e21dcf97146c2d9bbcb1278411c1
)

#add_library(mschol::mschol ALIAS mschol)
target_include_directories(mschol SYSTEM PUBLIC ${mschol_SOURCE_DIR}/src)