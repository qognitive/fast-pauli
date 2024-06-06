#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "fast_pauli.hpp"
#include <doctest/doctest.h>

TEST_CASE("empty") {
  // Simple test
  {
    std::vector<float> v_raw;
    std::mdspan v = fast_pauli::empty<float, 3>(v_raw, {3, 9, 19});

    CHECK(v.extent(0) == 3);
    CHECK(v.extent(1) == 9);
    CHECK(v.extent(2) == 19);
  }

  // Complex test
  {
    std::vector<std::complex<double>> v_raw;
    std::mdspan v =
        fast_pauli::empty<std::complex<double>, 3>(v_raw, {3, 9, 19});

    CHECK(v.extent(0) == 3);
    CHECK(v.extent(1) == 9);
    CHECK(v.extent(2) == 19);
  }
}

TEST_CASE("zeros") {
  // Simple test
  {
    std::vector<float> v_raw;
    std::mdspan v = fast_pauli::zeros<float, 3>(v_raw, {3, 9, 19});
    CHECK(v.extent(0) == 3);
    CHECK(v.extent(1) == 9);
    CHECK(v.extent(2) == 19);

    for (size_t i = 0; i < v.extent(0); ++i) {
      for (size_t j = 0; j < v.extent(1); ++j) {
        for (size_t k = 0; k < v.extent(2); ++k) {
          CHECK(v(i, j, k) == 0);
        }
      }
    }
  }

  // Complex test
  {
    std::vector<std::complex<double>> v_raw;
    std::mdspan v =
        fast_pauli::zeros<std::complex<double>, 3>(v_raw, {3, 9, 19});
    CHECK(v.extent(0) == 3);
    CHECK(v.extent(1) == 9);
    CHECK(v.extent(2) == 19);

    for (size_t i = 0; i < v.extent(0); ++i) {
      for (size_t j = 0; j < v.extent(1); ++j) {
        for (size_t k = 0; k < v.extent(2); ++k) {
          CHECK(v(i, j, k) == std::complex<double>(0));
        }
      }
    }
  }
}

TEST_CASE("rand") {
  {
    std::vector<float> v_raw;
    std::mdspan v = fast_pauli::rand<float, 3>(v_raw, {3, 9, 19});
    CHECK(v.extent(0) == 3);
    CHECK(v.extent(1) == 9);
    CHECK(v.extent(2) == 19);
  }

  {
    std::vector<std::complex<double>> v_raw;
    std::mdspan v =
        fast_pauli::rand<std::complex<double>, 3>(v_raw, {3, 9, 19});
    CHECK(v.extent(0) == 3);
    CHECK(v.extent(1) == 9);
    CHECK(v.extent(2) == 19);
  }
}