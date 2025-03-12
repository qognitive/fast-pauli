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

#include <nanobind/nanobind.h>

#include "__nb_helpers.hpp"
#include "__pauli_bindings.hpp"
#include "__pauli_op_bindings.hpp"
#include "__pauli_string_bindings.hpp"
#include "__summed_pauli_op_bindings.hpp"
#include "__types.hpp"

/*
Python Bindings for PauliOp
*/

NB_MODULE(_fast_pauli, m)
{

    // TODO init default threading behavior for the module
    // TODO give up GIL when calling into long-running C++ code
    // TODO != and == operators for our Pauli structures
    using float_type = double;

    init_pauli_bindings(m);
    init_paulistring_bindings(m);
    init_pauliop_bindings(m);
    init_summed_pauli_op_bindings(m);

    //
    // Helpers
    //
    auto helpers_m = m.def_submodule("helpers");
    helpers_m.def("get_nontrivial_paulis", &fp::get_nontrivial_paulis, "weight"_a,
                  R"%(Get all nontrivial Pauli strings up to a given weight.

Parameters
----------
weight : int
    Maximum weight of Pauli strings to return

Returns
-------
List[str]
    List of PauliStrings as strings
)%");

    helpers_m.def("calculate_pauli_strings", &fp::calculate_pauli_strings, "n_qubits"_a, "weight"_a,
                  R"%(Calculate all Pauli strings for a given weight.

Parameters
----------
n_qubits : int
    Number of qubits
weight : int
    Weight of Pauli strings to return

Returns
-------
List[PauliString]
    List of PauliStrings
)%");

    helpers_m.def("calculate_pauli_strings_max_weight", &fp::calculate_pauli_strings_max_weight, "n_qubits"_a,
                  "weight"_a,
                  R"%(Calculate all Pauli strings up to and including a given weight.

Parameters
----------
n_qubits : int
    Number of qubits
weight : int
    Maximum weight of Pauli strings to return

Returns
-------
List[PauliString]
    List of PauliStrings
)%");

    helpers_m.def("pauli_string_sparse_repr", &fp::get_sparse_repr<float_type>, "paulis"_a,
                  R"%(Get a sparse representation of a list of Pauli strings.

Parameters
----------
paulis : List[PauliString]
    List of PauliStrings

Returns
-------
List[Tuple[int, int]]
    List of tuples representing the Pauli string in a sparse format
)%")

        ;
}