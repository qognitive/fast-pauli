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
#include <fmt/core.h>
#include <fmt/ranges.h>

#include "fast_pauli.hpp"

using namespace fast_pauli;

TEST_CASE("get nontrivial paulis")
{
    {
        auto res = get_nontrivial_paulis(0);
        CHECK(res.size() == 0);
    }

    {
        auto res = get_nontrivial_paulis(1);
        CHECK(res.size() == 3);
        CHECK(res[0] == "X");
        CHECK(res[1] == "Y");
        CHECK(res[2] == "Z");
    }

    {
        auto res = get_nontrivial_paulis(2);
        CHECK(res.size() == 9);
        CHECK(res[0] == "XX");
        CHECK(res[1] == "XY");
        CHECK(res[2] == "XZ");
        CHECK(res[3] == "YX");
        CHECK(res[4] == "YY");
        CHECK(res[5] == "YZ");
        CHECK(res[6] == "ZX");
        CHECK(res[7] == "ZY");
        CHECK(res[8] == "ZZ");
    }

    {
        auto res = get_nontrivial_paulis(3);
        CHECK(res.size() == 27);
        CHECK(res[0] == "XXX");
        CHECK(res[1] == "XXY");
        CHECK(res[2] == "XXZ");
        CHECK(res[3] == "XYX");
        CHECK(res[4] == "XYY");
        CHECK(res[5] == "XYZ");
        CHECK(res[6] == "XZX");
        CHECK(res[7] == "XZY");
        CHECK(res[8] == "XZZ");
        CHECK(res[9] == "YXX");
        CHECK(res[10] == "YXY");
        CHECK(res[11] == "YXZ");
        CHECK(res[12] == "YYX");
        CHECK(res[13] == "YYY");
        CHECK(res[14] == "YYZ");
        CHECK(res[15] == "YZX");
        CHECK(res[16] == "YZY");
        CHECK(res[17] == "YZZ");
        CHECK(res[18] == "ZXX");
        CHECK(res[19] == "ZXY");
        CHECK(res[20] == "ZXZ");
        CHECK(res[21] == "ZYX");
        CHECK(res[22] == "ZYY");
        CHECK(res[23] == "ZYZ");
        CHECK(res[24] == "ZZX");
        CHECK(res[25] == "ZZY");
        CHECK(res[26] == "ZZZ");
    }
}

TEST_CASE("idx combinations")
{
    {
        auto res = idx_combinations(4, 1);
        CHECK(res.size() == 4);
        CHECK(res[0] == std::vector<size_t>{0});
        CHECK(res[1] == std::vector<size_t>{1});
        CHECK(res[2] == std::vector<size_t>{2});
        CHECK(res[3] == std::vector<size_t>{3});
    }

    {
        auto res = idx_combinations(4, 2);
        CHECK(res.size() == 6);
        CHECK(res[0] == std::vector<size_t>{0, 1});
        CHECK(res[1] == std::vector<size_t>{0, 2});
        CHECK(res[2] == std::vector<size_t>{0, 3});
        CHECK(res[3] == std::vector<size_t>{1, 2});
        CHECK(res[4] == std::vector<size_t>{1, 3});
        CHECK(res[5] == std::vector<size_t>{2, 3});
    }
}

TEST_CASE("calculate pauli strings")
{
    {
        auto res = calcutate_pauli_strings(4, 0);
        CHECK(res.size() == 1);
        CHECK(res[0] == PauliString("IIII"));
    }

    {
        auto res = calcutate_pauli_strings(2, 1);
        CHECK(res.size() == 6);
        CHECK(res[0] == PauliString("XI"));
        CHECK(res[1] == PauliString("IX"));
        CHECK(res[2] == PauliString("YI"));
        CHECK(res[3] == PauliString("IY"));
        CHECK(res[4] == PauliString("ZI"));
        CHECK(res[5] == PauliString("IZ"));
    }

    {
        auto res = calcutate_pauli_strings(4, 2);
        CHECK(res.size() == 54);
        CHECK(res[0] == PauliString("XXII"));
        CHECK(res[1] == PauliString("XIXI"));
        CHECK(res[53] == PauliString("IIZZ"));
    }
}

TEST_CASE("calculate pauli string max weight")
{
    {
        auto res = calculate_pauli_strings_max_weight(4, 0);
        CHECK(res.size() == 1);
        CHECK(res[0] == PauliString("IIII"));
    }

    {
        auto res = calculate_pauli_strings_max_weight(2, 1);
        CHECK(res.size() == 7);
        CHECK(res[0] == PauliString("II"));
        CHECK(res[1] == PauliString("XI"));
        CHECK(res[2] == PauliString("IX"));
        CHECK(res[3] == PauliString("YI"));
        CHECK(res[4] == PauliString("IY"));
        CHECK(res[5] == PauliString("ZI"));
        CHECK(res[6] == PauliString("IZ"));
    }

    {
        auto res = calculate_pauli_strings_max_weight(4, 2);
        CHECK(res.size() == 67);
    }

    {
        auto res = calculate_pauli_strings_max_weight(12, 2);
        CHECK(res.size() == 631);
    }
}