set(mas_test_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/../main.cpp
    test_bsr_matrix.cpp
    test_cusparse_bsr.cpp
    test_inner_product.cpp
    test_inverse.cpp
    test_mas_packed_inverse.cpp
)

add_executable(mas_utils_test ${mas_test_sources})

target_link_libraries(mas_utils_test PUBLIC Catch2::Catch2)
target_link_libraries(mas_utils_test PUBLIC polysolve::linear)

if(POLYSOLVE_WITH_SANITIZERS)
    add_sanitizers(mas_utils_test)
endif()

foreach(source IN ITEMS ${mas_test_sources})
    source_group("tests" FILES "${source}")
endforeach()

set(PARSE_CATCH_TESTS_ADD_TO_CONFIGURE_DEPENDS ON)
catch_discover_tests(mas_utils_test)
