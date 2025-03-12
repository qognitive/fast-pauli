#ifndef __FP_SUMMED_PAULI_OP_BINDINGS_HPP
#define __FP_SUMMED_PAULI_OP_BINDINGS_HPP
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
#include "__types.hpp"

namespace fp = fast_pauli;

/*
Python Bindings for PauliOp
*/

void init_summed_pauli_op_bindings(nb::module_ &m)
{

    //

    nb::class_<fp::SummedPauliOp<float_type>>(m, "SummedPauliOp")
        // Constructors
        // See
        // https://nanobind.readthedocs.io/en/latest/api_core.html#_CPPv4IDpEN8nanobind4initE
        .def(nb::init<>())
        .def(
            "__init__",
            [](fp::SummedPauliOp<float_type> *new_obj, std::vector<fp::PauliString> const &pauli_strings,
               nb::ndarray<cfloat_t> coeffs) {
                auto coeffs_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 2>(coeffs);
                new (new_obj) fp::SummedPauliOp<float_type>(pauli_strings, coeffs_mdspan);
            },
            "pauli_strings"_a, "coeffs"_a, R"%(Initialize SummedPauliOp from PauliStrings and coefficients.

Parameters
----------
pauli_strings : List[PauliString]
    List of PauliStrings to use in the SummedPauliOp (n_pauli_strings,)
coeffs : np.ndarray
    Array of coefficients corresponding to the PauliStrings (n_pauli_strings, n_operators)

Returns
-------
SummedPauliOp
    New SummedPauliOp instance
)%")
        .def(
            "__init__",
            [](fp::SummedPauliOp<float_type> *new_obj, std::vector<std::string> &pauli_strings,
               nb::ndarray<cfloat_t> coeffs) {
                //
                auto coeffs_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 2>(coeffs);

                new (new_obj) fp::SummedPauliOp<float_type>(pauli_strings, coeffs_mdspan);
            },
            "pauli_strings"_a, "coeffs"_a, R"%(Initialize SummedPauliOp from PauliStrings and coefficients.

Parameters
----------
pauli_strings : List[str]
    List of PauliStrings to use in the SummedPauliOp (n_pauli_strings,)
coeffs : np.ndarray
    Array of coefficients corresponding to the PauliStrings (n_pauli_strings, n_operators)

Returns
-------
SummedPauliOp
    New SummedPauliOp instance
)%")

        .def_prop_ro("dim", &fp::SummedPauliOp<float_type>::dim,
                     R"%(Return the Hilbert space dimension of the SummedPauliOp.

Returns
-------
int
    Hilbert space dimension
)%")
        .def_prop_ro("n_operators", &fp::SummedPauliOp<float_type>::n_operators,
                     R"%(Return the number of Pauli operators in the SummedPauliOp.

Returns
-------
int
    Number of operators
)%")
        .def_prop_ro("n_pauli_strings", &fp::SummedPauliOp<float_type>::n_pauli_strings,
                     R"%(Return the number of PauliStrings in the SummedPauliOp.

Returns
-------
int
    Number of PauliStrings
)%")
        .def_prop_rw(
            "coeffs",
            [](fp::SummedPauliOp<float_type> const &self) {
                std::vector<cfloat_t> coeffs_transposed(self.coeffs.size());
                auto coeffs_t = std::mdspan(coeffs_transposed.data(), self.coeffs.extent(1), self.coeffs.extent(0));

                for (size_t i = 0; i < self.coeffs.extent(0); i++)
                    for (size_t j = 0; j < self.coeffs.extent(1); j++)
                        coeffs_t(j, i) = self.coeffs(i, j);

                return fp::__detail::owning_ndarray_from_mdspan<cfloat_t, 2>(coeffs_t);
            },
            [](fp::SummedPauliOp<float_type> &self, nb::ndarray<cfloat_t> coeffs_new) {
                if (coeffs_new.ndim() != 2 or coeffs_new.shape(0) != self.n_operators() or
                    coeffs_new.shape(1) != self.n_pauli_strings())
                    throw std::invalid_argument(
                        "The shape of provided coeffs must match the number of operators and PauliStrings");

                auto coeffs_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 2>(coeffs_new);
                for (size_t i = 0; i < self.coeffs.extent(0); i++)
                    for (size_t j = 0; j < self.coeffs.extent(1); j++)
                        self.coeffs(i, j) = coeffs_mdspan(j, i);
            },
            nb::rv_policy::automatic, R"%(Getter and setter for coefficients.

Returns
-------
np.ndarray
    Array of coefficients corresponding with shape (n_operators, n_pauli_strings)
)%")
        .def_prop_ro(
            "pauli_strings", [](fp::SummedPauliOp<float_type> const &self) { return self.pauli_strings; },
            "List[PauliString]: The list of PauliString objects corresponding to coefficients in SummedPauliOp")
        .def_prop_ro(
            "pauli_strings_as_str",
            [](fp::SummedPauliOp<float_type> const &self) {
                std::vector<std::string> strings(self.n_pauli_strings());
                std::transform(self.pauli_strings.begin(), self.pauli_strings.end(), strings.begin(),
                               [](fp::PauliString const &ps) { return fmt::format("{}", ps); });
                return strings;
            },
            "List[str]: The list of Pauli Strings representations corresponding to coefficients from SummedPauliOp")

        .def(
            "apply",
            [](fp::SummedPauliOp<float_type> const &self, nb::ndarray<cfloat_t> states) {
                if (states.ndim() == 1)
                {
                    auto states_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 1>(states);
                    auto states_mdspan_2d = std::mdspan(states_mdspan.data_handle(), states_mdspan.extent(0), 1);
                    auto new_states = fp::__detail::owning_ndarray_like_mdspan<cfloat_t, 1>(states_mdspan);
                    auto new_states_mdspan = std::mdspan(new_states.data(), new_states.size(), 1);
                    self.apply(std::execution::par, new_states_mdspan, states_mdspan_2d);
                    return new_states;
                }
                else if (states.ndim() == 2)
                {
                    auto states_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 2>(states);
                    auto new_states = fp::__detail::owning_ndarray_like_mdspan<cfloat_t, 2>(states_mdspan);
                    auto new_states_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 2>(new_states);
                    self.apply(std::execution::par, new_states_mdspan, states_mdspan);
                    return new_states;
                }
                else
                {
                    throw std::invalid_argument(
                        fmt::format("apply: expected 1 or 2 dimensions, got {}", states.ndim()));
                }
            },
            "states"_a,
            R"%(Apply the SummedPauliOp to a batch of states.

.. math::
    \big(\sum_k \sum_i h_{ik} \mathcal{\hat{P}}_i \big) \ket{\psi_t}

Parameters
----------
states : np.ndarray
    The original state(s) represented as 2D numpy array (n_dims, n_states) for batched calculation.

Returns
-------
np.ndarray
    New state(s) in a form of 2D numpy array (n_dims, n_states) according to the shape of input states
)%")

        .def(
            "apply_weighted",
            [](fp::SummedPauliOp<float_type> const &self, nb::ndarray<cfloat_t> states, nb::ndarray<float_type> data) {
                if (states.ndim() == 1)
                {
                    auto states_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 1>(states);
                    auto states_mdspan_2d = std::mdspan(states_mdspan.data_handle(), states_mdspan.size(), 1);
                    auto data_mdspan = fp::__detail::ndarray_to_mdspan<float_type, 1>(data);
                    auto data_mdspan_2d = std::mdspan(data_mdspan.data_handle(), data_mdspan.size(), 1);
                    auto new_states = fp::__detail::owning_ndarray_like_mdspan<cfloat_t, 1>(states_mdspan);
                    auto new_states_mdspan = std::mdspan(new_states.data(), new_states.size(), 1);
                    self.apply_weighted(std::execution::par, new_states_mdspan, states_mdspan_2d, data_mdspan_2d);
                    return new_states;
                }
                else if (states.ndim() == 2)
                {
                    auto states_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 2>(states);
                    auto data_mdspan = fp::__detail::ndarray_to_mdspan<float_type, 2>(data);
                    auto new_states = fp::__detail::owning_ndarray_like_mdspan<cfloat_t, 2>(states_mdspan);
                    auto new_states_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 2>(new_states);
                    self.apply_weighted(std::execution::par, new_states_mdspan, states_mdspan, data_mdspan);
                    return new_states;
                }
                else
                {
                    throw std::invalid_argument(
                        fmt::format("apply_weighted: expected 1 or 2 dimensions for states, got {}", states.ndim()));
                }
            },
            "states"_a, "data"_a,
            R"%(Apply the SummedPauliOp to a batch of states with corresponding weights.

.. math::
    \big(\sum_k x_{tk} \sum_i h_{ik} \mathcal{\hat{P}}_i \big) \ket{\psi_t}

Parameters
----------
states : np.ndarray
    The original state(s) represented as 2D numpy array (n_dims, n_states) for batched calculation.
data : np.ndarray
    The data to weight the operators corresponding to the states (n_operators, n_states)

Returns
-------
np.ndarray
    New state(s) in a form of 2D numpy array (n_dims, n_states) according to the shape of input states
)%")
        .def(
            "expectation_value",
            [](fp::SummedPauliOp<float_type> const &self, nb::ndarray<cfloat_t> states) {
                if (states.ndim() == 1)
                {
                    auto states_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 1>(states);
                    auto states_mdspan_2d = std::mdspan(states_mdspan.data_handle(), states_mdspan.size(), 1);
                    std::array<size_t, 1> out_shape = {self.n_operators()};
                    auto expected_vals_out = fp::__detail::owning_ndarray_from_shape<cfloat_t, 1>(out_shape);
                    auto expected_vals_out_mdspan = std::mdspan(expected_vals_out.data(), expected_vals_out.size(), 1);
                    self.expectation_value(std::execution::par, expected_vals_out_mdspan, states_mdspan_2d);
                    return expected_vals_out;
                }
                else if (states.ndim() == 2)
                {
                    auto states_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 2>(states);
                    std::array<size_t, 2> out_shape = {self.n_operators(), states_mdspan.extent(1)};
                    auto expected_vals_out = fp::__detail::owning_ndarray_from_shape<cfloat_t, 2>(out_shape);
                    auto expected_vals_out_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 2>(expected_vals_out);
                    self.expectation_value(std::execution::par, expected_vals_out_mdspan, states_mdspan);
                    return expected_vals_out;
                }
                else
                {
                    throw std::invalid_argument(
                        fmt::format("expectation_value: expected 1 or 2 dimensions for states, got {}", states.ndim()));
                }
            },
            "states"_a,
            R"%(Calculate expectation value(s) for a given batch of states.

.. math::
    \bra{\psi_t} \big(\sum_k \sum_i h_{ik} \mathcal{\hat{P}}_i \big) \ket{\psi_t}

Parameters
----------
states : np.ndarray
    The state(s) represented as 2D numpy array (n_operators, n_states) for batched calculation.

Returns
-------
np.ndarray
    Expectation value(s) in a form of 2D numpy array (n_operators, n_states) according to the shape of input states
)%")
        .def(
            "to_tensor",
            [](fp::SummedPauliOp<float_type> const &self) {
                auto dense_op =
                    fp::__detail::owning_ndarray_from_shape<cfloat_t, 3>({self.n_operators(), self.dim(), self.dim()});
                self.to_tensor(fp::__detail::ndarray_to_mdspan<cfloat_t, 3>(dense_op));
                return dense_op;
            },
            R"%(Returns a dense representation of SummedPauliOp.

Returns
-------
np.ndarray
    3D numpy array of complex numbers with a shape of (n_operators, 2^n_qubits, 2^n_qubits)
)%")
        .def(
            "clone",
            [](fp::SummedPauliOp<float_type> const &self) {
                return fp::SummedPauliOp<float_type>(self.pauli_strings, self.coeffs_raw);
            },
            R"%(Returns a copy of the SummedPauliOp object.

Returns
-------
SummedPauliOp
    A copy of the SummedPauliOp object
)%")
        .def("split", &fp::SummedPauliOp<float_type>::split,
             R"%(Returns all components of the SummedPauliOp expressed as a vector of PauliOps.

Returns
-------
List[fp.PauliOp]
    Components of the SummedPauliOp object
)%")
        .def("square", &fp::SummedPauliOp<float_type>::square, R"%(Square the SummedPauliOp.

Returns
-------
SummedPauliOp
    New SummedPauliOp instance
)%")
        .def("__getstate__",
             [](fp::SummedPauliOp<float_type> const &self) {
                 return std::make_tuple(self.pauli_strings, self.coeffs_raw);
             })
        .def("__setstate__", [](fp::SummedPauliOp<float_type> &self,
                                std::tuple<std::vector<fp::PauliString>, std::vector<cfloat_t>> state) {
            new (&self) fp::SummedPauliOp<float_type>(std::get<0>(state), std::get<1>(state));
        });
    //
    ;
}

#endif
