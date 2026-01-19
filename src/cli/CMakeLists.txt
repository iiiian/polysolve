set(SOURCES
    main.cpp
    mmio.c
)

find_package(argparse REQUIRED)

add_executable(polysolve_cli ${SOURCES})

target_link_libraries(polysolve_cli PRIVATE polysolve::linear argparse::argparse)
target_include_directories(polysolve_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(polysolve_cli PRIVATE polysolve::warnings)


