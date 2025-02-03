if(TARGET mschol::mschol)
    return()
endif()

message(STATUS "Third-party: creating target 'mschol::mschol'")

include(CPM)
CPMAddPackage(
    NAME mschol
    GIT_REPOSITORY https://gitlab.inria.fr/geomerix/ichol
    GIT_TAG 580f7555a3b10e4d9cb5cd79bd8de05adbe56527
)

add_library(mschol::mschol ALIAS mschol)
target_include_directories(mschol SYSTEM PUBLIC ${mschol_SOURCE_DIR}/src)