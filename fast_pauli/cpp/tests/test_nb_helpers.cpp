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

#include "__nb_helpers.hpp"
#include "fast_pauli.hpp"

namespace fp = fast_pauli;
namespace fp_det = fast_pauli::__detail;

TEST_CASE("owning_ndarray_from_mdspan")
{
    SUBCASE("single item")
    {
        std::vector<float> storage;
        auto view = fp::zeros<float, 1>(storage, {1});
        auto ndarray = fp_det::owning_ndarray_from_mdspan<float, 1>(view);

        CHECK(ndarray.size() == 1);
        CHECK(ndarray.ndim() == 1);
        CHECK(ndarray.shape(0) == 1);
    }

    SUBCASE("rank 2")
    {
        using complex_t = std::complex<double>;

        std::vector<complex_t> storage;
        std::mdspan view = fast_pauli::empty<complex_t, 2>(storage, {3, 3});
        std::iota(storage.begin(), storage.end(), 1);
        auto ndarray = fp_det::owning_ndarray_from_mdspan<complex_t, 2>(view);

        CHECK(ndarray.size() == 9);
        CHECK(ndarray.shape(0) == 3);
        CHECK(ndarray.shape(1) == 3);
        CHECK(std::equal(storage.begin(), storage.end(), ndarray.data()));
    }
}
