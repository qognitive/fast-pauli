/**
 * This code is part of Fast Pauli.
 *
 * (C) Copyright Qognitive Inc 2024.
 *
 * This code is licensed under the BSD 2-Clause License. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.


 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest/doctest.h>

#include "fast_pauli.hpp"

TEST_CASE("empty")
{
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
        std::mdspan v = fast_pauli::empty<std::complex<double>, 3>(v_raw, {3, 9, 19});

        CHECK(v.extent(0) == 3);
        CHECK(v.extent(1) == 9);
        CHECK(v.extent(2) == 19);
    }
}

TEST_CASE("zeros")
{
    // Simple test
    {
        std::vector<float> v_raw;
        std::mdspan v = fast_pauli::zeros<float, 3>(v_raw, {3, 9, 19});
        CHECK(v.extent(0) == 3);
        CHECK(v.extent(1) == 9);
        CHECK(v.extent(2) == 19);

        for (size_t i = 0; i < v.extent(0); ++i)
        {
            for (size_t j = 0; j < v.extent(1); ++j)
            {
                for (size_t k = 0; k < v.extent(2); ++k)
                {
                    CHECK(v(i, j, k) == 0);
                }
            }
        }
    }

    // Complex test
    {
        std::vector<std::complex<double>> v_raw;
        std::mdspan v = fast_pauli::zeros<std::complex<double>, 3>(v_raw, {3, 9, 19});
        CHECK(v.extent(0) == 3);
        CHECK(v.extent(1) == 9);
        CHECK(v.extent(2) == 19);

        for (size_t i = 0; i < v.extent(0); ++i)
        {
            for (size_t j = 0; j < v.extent(1); ++j)
            {
                for (size_t k = 0; k < v.extent(2); ++k)
                {
                    CHECK(v(i, j, k) == std::complex<double>(0));
                }
            }
        }
    }
}

TEST_CASE("rand")
{
    {
        std::vector<float> v_raw;
        std::mdspan v = fast_pauli::rand<float, 3>(v_raw, {3, 9, 19});
        CHECK(v.extent(0) == 3);
        CHECK(v.extent(1) == 9);
        CHECK(v.extent(2) == 19);
    }

    {
        std::vector<std::complex<double>> v_raw;
        std::mdspan v = fast_pauli::rand<std::complex<double>, 3>(v_raw, {3, 9, 19});
        CHECK(v.extent(0) == 3);
        CHECK(v.extent(1) == 9);
        CHECK(v.extent(2) == 19);
    }
}