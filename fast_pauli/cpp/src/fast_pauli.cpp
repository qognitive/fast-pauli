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

/*
Python Bindings for PauliOp
*/

NB_MODULE(_fast_pauli, m)
{

    init_pauli_bindings(m);
    init_paulistring_bindings(m);
    init_pauliop_bindings(m);
    init_summed_pauli_op_bindings(m);
}