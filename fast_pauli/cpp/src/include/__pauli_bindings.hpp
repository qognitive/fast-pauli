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

#include "fast_pauli.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>

#include <vector>

#include "__nb_helpers.hpp"

namespace fp = fast_pauli;

/*
Python Bindings for fp::Pauli
*/

void init_pauli_bindings(nb::module_ &m)
{
    // TODO init default threading behavior for the module
    // TODO give up GIL when calling into long-running C++ code
    // TODO != and == operators for our Pauli structures
    using float_type = double;
    using cfloat_t = std::complex<float_type>;

    nb::class_<fp::Pauli>(
        m, "Pauli",
        R"%(A class for efficient representation of a :math:`2 \times 2` Pauli Matrix :math:`\sigma_i \in \{ I, X, Y, Z \}`)%")
        // Constructors
        .def(nb::init<>(), "Default constructor to initialize with identity matrix.")
        .def(nb::init<int const>(), "code"_a,
             R"%(Constructor given a numeric code.

Parameters
----------
code : int
    Numerical label of type int for corresponding Pauli matrix :math:`0: I, 1: X, 2: Y, 3: Z`
)%")
        .def(nb::init<char const>(), "symbol"_a,
             R"%(Constructor given Pauli matrix symbol.

Parameters
----------
symbol : str
    Character label of type str corresponding to one of the Pauli Matrix symbols :math:`I, X, Y, Z`
)%")
        // Methods
        .def(
            "__matmul__", [](fp::Pauli const &self, fp::Pauli const &rhs) { return self * rhs; }, nb::is_operator(),
            R"%(Returns matrix product of two Paulis as a tuple of phase and new Pauli object.

Parameters
----------
rhs : Pauli
    Right hand side Pauli object

Returns
-------
tuple[complex, Pauli]
    Phase and resulting Pauli object
)%")
        .def(
            "to_tensor",
            [](fp::Pauli const &self) {
                auto dense_pauli = fp::__detail::owning_ndarray_from_shape<cfloat_t, 2>({2, 2});
                self.to_tensor(fp::__detail::ndarray_to_mdspan<cfloat_t, 2>(dense_pauli));
                return dense_pauli;
            },
            R"%(Returns a dense representation of Pauli object as a :math:`2 \times 2` matrix.

Returns
-------
np.ndarray
    2D numpy array of complex numbers
)%")
        .def(
            "clone", [](fp::Pauli const &self) { return fp::Pauli(self); },
            R"%(Returns a copy of the Pauli object.

Returns
-------
Pauli
    A copy of the Pauli object
)%")
        .def(
            "__str__", [](fp::Pauli const &self) { return fmt::format("{}", self); },
            R"%(Returns a string representation of Pauli matrix.

Returns
-------
str
    One of :math:`I, X, Y, Z`, a single character string representing a Pauli Matrix
)%");
}