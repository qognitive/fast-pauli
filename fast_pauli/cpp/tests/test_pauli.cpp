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

#include <unordered_map>

#include "fast_pauli.hpp"

using namespace fast_pauli;

// Need to add a specialization so we can use Pauli as a key in a hash map
template <> struct std::hash<Pauli>
{
    std::size_t operator()(Pauli const &p) const
    {
        return std::hash<uint8_t>{}(p.code);
    }
};

TEST_CASE("test the limits of pauli code")
{
    Pauli p0{0};
    Pauli p1{1};
    Pauli p2{2};
    Pauli p3{3};
    // Pauli p4{4};  // Implementation defined behavior

    CHECK(p0.code == 0);
    CHECK(p1.code == 1);
    CHECK(p2.code == 2);
    CHECK(p3.code == 3);

    fmt::print("Size of Pauli: {}\n", sizeof(Pauli));
}

// Helper function
template <> struct std::hash<std::pair<Pauli, Pauli>>
{
    std::size_t operator()(auto const &pp) const
    {
        auto [p, q] = pp;
        return std::hash<uint8_t>{}(p.code) + 8 * std::hash<uint8_t>{}(q.code);
    }
};

TEST_CASE("check pauli mult")
{
    Pauli I{0};
    Pauli X{1};
    Pauli Y{2};
    Pauli Z{3};

    std::unordered_map<std::pair<Pauli, Pauli>, std::pair<std::complex<double>, Pauli>> cayley_table = {
        {{I, I}, {1, I}},
        {{I, X}, {1, X}},
        {{I, Y}, {1, Y}},
        {{I, Z}, {1, Z}},
        //
        {{X, I}, {1, X}},
        {{X, X}, {1, I}},
        {{X, Y}, {1i, Z}},
        {{X, Z}, {-1i, Y}},
        //
        {{Y, I}, {1, Y}},
        {{Y, X}, {-1i, Z}},
        {{Y, Y}, {1, I}},
        {{Y, Z}, {1i, X}},
        //
        {{Z, I}, {1, Z}},
        {{Z, X}, {1i, Y}},
        {{Z, Y}, {-1i, X}},
        {{Z, Z}, {1, I}},
    };

    for (auto p : {I, X, Y, Z})
    {
        for (auto q : {I, X, Y, Z})
        {
            auto [phase, res] = p * q;
            auto [true_phase, true_res] = cayley_table.at({p, q});
            CHECK(phase == true_phase);
            CHECK(res == true_res);

            fmt::print("{} * {} = ({} + {}i) * {}\n", p, q, phase.real(), phase.imag(), res);
        }
    }
}
