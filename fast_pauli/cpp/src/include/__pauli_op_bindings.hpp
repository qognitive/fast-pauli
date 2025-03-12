#ifndef __FP_PAULI_OP_BINDINGS_HPP
#define __FP_PAULI_OP_BINDINGS_HPP
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

void init_pauliop_bindings(nb::module_ &m)
{

    nb::class_<fp::PauliOp<float_type>>(
        m, "PauliOp",
        R"%(A class representation for a Pauli Operator :math:`A` (i.e. a weighted sum of Pauli Strings)

.. math::
    A = \sum_j h_j \mathcal{\hat{P}}_j

    \mathcal{\hat{P}} = \bigotimes_i \sigma_i \quad h_j \in \mathbb{C}
)%")
        // Constructors
        .def(nb::init<>(), "Default constructor to initialize strings and coefficients with empty arrays.")
        .def(nb::init<std::vector<std::string> const &>(), "pauli_strings"_a,
             R"%(Construct a PauliOp from a list of strings and default corresponding coefficients to ones.

Parameters
----------
pauli_strings : List[str]
    List of Pauli Strings as simple `str`. Each string should be composed of characters :math:`I, X, Y, Z` and should have the same size
)%")
        .def(nb::init<std::vector<fp::PauliString>>(),
             R"%(Construct a PauliOp from a list of PauliString objects and default corresponding coefficients to ones.

Parameters
----------
pauli_strings : List[PauliString]
    List of PauliString objects.
)%")
        .def(
            "__init__",
            [](fp::PauliOp<float_type> *new_obj, nb::ndarray<cfloat_t> coeffs,
               std::vector<fp::PauliString> const &pauli_strings) {
                auto [coeffs_vec, _] = fp::__detail::ndarray_to_raw<cfloat_t, 1>(coeffs);
                new (new_obj) fp::PauliOp<float_type>(coeffs_vec, pauli_strings);
            },
            "coefficients"_a, "pauli_strings"_a,
            R"%(Construct a PauliOp from a list of PauliString objects and corresponding coefficients.

Parameters
----------
coefficients : np.ndarray
    Array of coefficients corresponding to Pauli strings.
pauli_strings : List[PauliString]
    List of PauliString objects.
)%")
        .def(
            "__init__",
            [](fp::PauliOp<float_type> *new_obj, std::vector<cfloat_t> coeffs_vec,
               std::vector<fp::PauliString> const &pauli_strings) {
                new (new_obj) fp::PauliOp<float_type>(coeffs_vec, pauli_strings);
            },
            "coefficients"_a, "pauli_strings"_a,
            R"%(Construct a PauliOp from a list of PauliString objects and corresponding coefficients.

Parameters
----------
coefficients : List[complex]
    List of coefficients corresponding to Pauli strings.
pauli_strings : List[PauliString]
    List of PauliString objects.
)%")
        .def(
            "__init__",
            [](fp::PauliOp<float_type> *new_obj, std::vector<cfloat_t> coeffs_vec,
               std::vector<std::string> const &strings) {
                std::vector<fp::PauliString> pauli_strings;
                std::transform(strings.begin(), strings.end(), std::back_inserter(pauli_strings),
                               [](std::string const &pauli) { return fp::PauliString(pauli); });
                new (new_obj) fp::PauliOp<float_type>(coeffs_vec, pauli_strings);
            },
            "coefficients"_a, "pauli_strings"_a,
            R"%(Construct a PauliOp from a list of strings and corresponding coefficients.

Parameters
----------
coefficients : np.ndarray
    Array of coefficients corresponding to Pauli strings.
pauli_strings : List[str]
    List of Pauli Strings as simple `str`. Each string should be composed of characters :math:`I, X, Y, Z` and should have the same size
)%")
        // TODO memory efficient implementations for inplace @= operators
        .def(
            "__matmul__",
            [](fp::PauliOp<float_type> const &self, fp::PauliOp<float_type> const &rhs) { return self * rhs; },
            nb::is_operator(),
            R"%(Efficient matrix multiplication of two Pauli Operators, leveraging their sparse structure.

Parameters
----------
rhs : PauliOp
    Right hand side PauliOp object

Returns
-------
PauliOp
    New PauliOp instance containing the product
)%")
        .def(
            "__matmul__", [](fp::PauliOp<float_type> const &self, fp::PauliString const &rhs) { return self * rhs; },
            nb::is_operator(),
            R"%(Efficient matrix multiplication of PauliOp with a PauliString on the right, leveraging their sparse structure.

Parameters
----------
rhs : PauliString
    Right hand side PauliString object

Returns
-------
PauliOp
    New PauliOp instance containing the product
)%")
        .def(
            "__rmatmul__", [](fp::PauliOp<float_type> const &self, fp::PauliString const &lhs) { return lhs * self; },
            nb::is_operator(),
            R"%(Efficient matrix multiplication of PauliOp with a PauliString on the left, leveraging their sparse structure.

Parameters
----------
rhs : PauliOp
    Left hand side PauliOp object

Returns
-------
PauliOp
    New PauliOp instance containing the product
)%")
        .def(
            "__mul__",
            [](fp::PauliOp<float_type> const &self, cfloat_t rhs) {
                fp::PauliOp<float_type> res_op(self);
                res_op.scale(rhs);
                return res_op;
            },
            nb::is_operator(),
            R"%(Scale Pauli Operator by a scalar value.

Parameters
----------
rhs : complex or float
    Right hand side scalar multiplier

Returns
-------
PauliOp
    New PauliOp instance containing the product
)%")
        .def(
            "__rmul__",
            [](fp::PauliOp<float_type> const &self, cfloat_t lhs) {
                fp::PauliOp<float_type> res_op(self);
                res_op.scale(lhs);
                return res_op;
            },
            nb::is_operator(),
            R"%(Scale Pauli Operator by a scalar value.

Parameters
----------
lhs : complex or float
    Left hand side scalar multiplier

Returns
-------
PauliOp
    New PauliOp instance containing the product
)%")
        .def(
            "__imul__",
            [](fp::PauliOp<float_type> &self, cfloat_t factor) {
                self.scale(factor);
                return self;
            },
            nb::is_operator(),
            R"%(Scale Pauli Operator inplace by a scalar value.

Parameters
----------
other : complex or float
    Scalar multiplier

Returns
-------
PauliOp
    Current PauliOp instance after scaling
)%")
        .def(
            "__add__",
            [](fp::PauliOp<float_type> const &lhs_op, fp::PauliOp<float_type> const &rhs_op) {
                fp::PauliOp<float_type> res_op(lhs_op);
                res_op.extend(rhs_op);
                return res_op;
            },
            nb::is_operator(),
            R"%(Returns the sum of two Pauli Operators.

Parameters
----------
rhs : PauliOp
    The other PauliOp object to add

Returns
-------
PauliOp
    New PauliOp instance holding the sum.
)%")
        .def(
            "__add__",
            [](fp::PauliOp<float_type> const &lhs_op, fp::PauliString const &rhs_str) {
                fp::PauliOp<float_type> res_op(lhs_op);
                res_op.extend(rhs_str, 1);
                return res_op;
            },
            nb::is_operator(),
            R"%(Returns the sum of Pauli Operator with Pauli String.

Parameters
----------
rhs : PauliString
    Right hand side PauliString object to add

Returns
-------
PauliOp
    New PauliOp instance holding the sum.
)%")
        .def(
            "__radd__",
            [](fp::PauliOp<float_type> const &self, fp::PauliString const &lhs_str) {
                fp::PauliOp<float_type> res_op(self);
                res_op.extend(lhs_str, 1);
                return res_op;
            },
            nb::is_operator(),
            R"%(Returns the sum of Pauli Operators with Pauli String.

Parameters
----------
lhs : PauliString
    Left hand side PauliString object to add

Returns
-------
PauliOp
    New PauliOp instance holding the sum.
)%")
        .def(
            "__iadd__",
            [](fp::PauliOp<float_type> &self, fp::PauliOp<float_type> const &other_op) {
                self.extend(other_op);
                return self;
            },
            nb::is_operator(),
            R"%(Performs inplace addition with other Pauli Operator.

Parameters
----------
other : PauliOp
    Pauli operator object to add

Returns
-------
PauliOp
    Current PauliOp instance after addition
)%")
        .def(
            "__iadd__",
            [](fp::PauliOp<float_type> &self, fp::PauliString const &other_str) {
                self.extend(other_str, 1);
                return self;
            },
            nb::is_operator(),
            R"%(Performs inplace addition with Pauli String.

Parameters
----------
other : PauliString
    Pauli string object to add

Returns
-------
PauliOp
    Current PauliOp instance after addition
)%")
        .def(
            "__sub__",
            [](fp::PauliOp<float_type> const &lhs_op, fp::PauliOp<float_type> const &rhs_op) {
                fp::PauliOp<float_type> res_op(lhs_op);
                res_op.extend(-rhs_op);
                return res_op;
            },
            nb::is_operator(),
            R"%(Returns the difference of two Pauli Operators.

Parameters
----------
rhs : PauliOp
    The other PauliOp object to subtract

Returns
-------
PauliOp
    New PauliOp instance holding the difference.
)%")
        .def(
            "__sub__",
            [](fp::PauliOp<float_type> const &lhs_op, fp::PauliString const &rhs_str) {
                fp::PauliOp<float_type> res_op(lhs_op);
                res_op.extend(rhs_str, -1);
                return res_op;
            },
            nb::is_operator(),
            R"%(Returns the difference of Pauli Operator with Pauli String.

Parameters
----------
rhs : PauliString
    Right hand side PauliString object to subtract

Returns
-------
PauliOp
    New PauliOp instance holding the difference.
)%")
        .def(
            "__rsub__",
            [](fp::PauliOp<float_type> const &self, fp::PauliString const &lhs_str) {
                fp::PauliOp<float_type> res_op(-self);
                res_op.extend(lhs_str, 1);
                return res_op;
            },
            nb::is_operator(),
            R"%(Returns the difference of Pauli Operators with Pauli String.

Parameters
----------
lhs : PauliString
    Left hand side PauliString object to subtract

Returns
-------
PauliOp
    New PauliOp instance holding the difference.
)%")
        .def(
            "__isub__",
            [](fp::PauliOp<float_type> &self, fp::PauliOp<float_type> const &other_op) {
                self.extend(-other_op);
                return self;
            },
            nb::is_operator(),
            R"%(Performs inplace subtraction with other Pauli Operator.

Parameters
----------
other : PauliOp
    Pauli operator object to subtract

Returns
-------
PauliOp
    Current PauliOp instance after subtraction
)%")
        .def(
            "__isub__",
            [](fp::PauliOp<float_type> &self, fp::PauliString const &other_str) {
                self.extend(other_str, -1);
                return self;
            },
            nb::is_operator(),
            R"%(Performs inplace subtraction with Pauli String.

Parameters
----------
other : PauliString
    Pauli string object to subtract

Returns
-------
PauliOp
    Current PauliOp instance after subtraction
)%")
        .def(
            "extend", [](fp::PauliOp<float_type> &self, fp::PauliOp<float_type> const &other) { self.extend(other); },
            "other"_a,
            R"%(Add another PauliOp to the current one by extending the internal summation with new terms.

Parameters
----------
other : PauliOp
    PauliOp object to extend the current one with
)%")
        .def(
            "extend",
            [](fp::PauliOp<float_type> &self, fp::PauliString const &other, cfloat_t multiplier, bool dedupe) {
                self.extend(other, multiplier, dedupe);
            },
            "other"_a, "multiplier"_a, "dedupe"_a = true,
            R"%(Add a Pauli String term with a corresponding coefficient to the summation inside PauliOp.

Parameters
----------
other : PauliString
    PauliString object to add to the summation
multiplier : complex
    Coefficient to apply to the PauliString
dedupe : bool
    Whether to deduplicate the set of PauliStrings
)%")

        // Getters
        .def_prop_ro("dim", &fp::PauliOp<float_type>::dim,
                     "int: The dimension of PauliStrings used to compose PauliOp :math:`2^n, n` - number of qubits")
        .def_prop_ro("n_qubits", &fp::PauliOp<float_type>::n_qubits, "int: The number of qubits in PauliOp")
        .def_prop_ro("n_pauli_strings", &fp::PauliOp<float_type>::n_pauli_strings,
                     "int: The number of PauliString terms in PauliOp")
        .def_prop_ro(
            "coeffs", [](fp::PauliOp<float_type> const &self) { return self.coeffs; },
            "List[complex]: The list of coefficients corresponding to Pauli strings")
        .def_prop_ro(
            "pauli_strings", [](fp::PauliOp<float_type> const &self) { return self.pauli_strings; },
            "List[PauliString]: The list of PauliString objects corresponding to coefficients in PauliOp")
        .def_prop_ro(
            "pauli_strings_as_str",
            [](fp::PauliOp<float_type> const &self) {
                //  return self.pauli_strings;
                std::vector<std::string> strings(self.n_pauli_strings());
                std::transform(self.pauli_strings.begin(), self.pauli_strings.end(), strings.begin(),
                               [](fp::PauliString const &ps) { return fmt::format("{}", ps); });
                return strings;
            },
            "List[str]: The list of PauliString representations corresponding to coefficients in PauliOp")

        // Methods
        .def(
            "scale", [](fp::PauliOp<float_type> &self, cfloat_t factor) { self.scale(factor); }, "factor"_a,
            R"%(Scale each individual term of Pauli Operator by a scalar value.

Parameters
----------
factor : complex or float
    Scalar multiplier
)%")
        .def(
            "scale",
            [](fp::PauliOp<float_type> &self, nb::ndarray<cfloat_t> factors) {
                auto factors_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 1>(factors);
                self.scale(factors_mdspan);
            },
            "factors"_a,
            R"%(Scale each individual term of Pauli Operator by a scalar value.

Parameters
----------
factors : np.ndarray
    Array of factors to scale each term with. The length of the array should match the number of Pauli strings in PauliOp
)%")
        .def(
            "apply",
            [](fp::PauliOp<float_type> const &self, nb::ndarray<cfloat_t> states) {
                if (states.ndim() == 1)
                {
                    auto states_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 1>(states);
                    auto new_states = fp::__detail::owning_ndarray_like_mdspan<cfloat_t, 1>(states_mdspan);
                    auto new_states_mdspan = std::mdspan(new_states.data(), new_states.size());
                    self.apply(std::execution::par, new_states_mdspan, states_mdspan);
                    return new_states;
                }
                else if (states.ndim() == 2)
                {
                    auto states_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 2>(states);
                    auto new_states = fp::__detail ::owning_ndarray_like_mdspan<cfloat_t, 2>(states_mdspan);
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
            R"%(Apply a Pauli Operator to a single dimensional state vector or a batch of states.

.. math::
    \big( \sum_j h_j \mathcal{\hat{P}}_j \big) \ket{\psi_t}

.. note::
    For batch mode it applies the PauliOp to each individual state separately.
    In this case, the input array is expected to have the shape of (n_dims, n_states) with states stored as columns.

Parameters
----------
states : np.ndarray
    The original state(s) represented as 1D (n_dims,) or 2D numpy array (n_dims, n_states) for batched calculation.
    Outer dimension must match the dimensionality of Pauli Operator.

Returns
-------
np.ndarray
    New state(s) in a form of 1D (n_dims,) or 2D numpy array (n_dims, n_states) according to the shape of input states
)%")
        .def(
            "expectation_value",
            [](fp::PauliOp<float_type> const &self, nb::ndarray<cfloat_t> states) {
                if (states.ndim() == 1)
                {
                    auto states_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 1>(states);
                    auto states_mdspan_2d = std::mdspan(states_mdspan.data_handle(), states_mdspan.extent(0), 1);
                    std::array<size_t, 1> out_shape = {1};
                    auto expected_vals_out = fp::__detail::owning_ndarray_from_shape<cfloat_t, 1>(out_shape);
                    auto expected_vals_out_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 1>(expected_vals_out);
                    self.expectation_value(std::execution::par, expected_vals_out_mdspan, states_mdspan_2d);
                    return expected_vals_out;
                }
                else if (states.ndim() == 2)
                {
                    auto states_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 2>(states);
                    std::array<size_t, 1> out_shape = {states_mdspan.extent(1)};
                    auto expected_vals_out = fp::__detail::owning_ndarray_from_shape<cfloat_t, 1>(out_shape);
                    auto expected_vals_out_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 1>(expected_vals_out);

                    self.expectation_value(std::execution::par, expected_vals_out_mdspan, states_mdspan);

                    return expected_vals_out;
                }
                else
                {
                    throw std::invalid_argument(
                        fmt::format("expectation_value: expected 1 or 2 dimensions, got {}", states.ndim()));
                }
            },
            "states"_a,
            R"%(Calculate expectation value(s) for a given single dimensional state vector or a batch of states.

.. math::
    \bra{\psi_t} \big( \sum_j h_j \mathcal{\hat{P}}_j \big) \ket{\psi_t}

.. note::
    For batch mode it computes the expectation value for each individual state separately.
    In this case, the input array is expected to have the shape of (n_dims, n_states) with states stored as columns.

Parameters
----------
states : np.ndarray
    The original state(s) represented as 1D (n_dims,) or 2D numpy array (n_dims, n_states) for batched calculation.
    Outer dimension must match the dimensionality of Pauli Operator.

Returns
-------
np.ndarray
    Expectation value(s) in the form of a 1D numpy array with a shape of (n_states,)
)%")
        .def(
            "to_tensor",
            [](fp::PauliOp<float_type> const &self) {
                auto dense_op = fp::__detail::owning_ndarray_from_shape<cfloat_t, 2>({self.dim(), self.dim()});
                self.to_tensor(fp::__detail::ndarray_to_mdspan<cfloat_t, 2>(dense_op));
                return dense_op;
            },
            R"%(Returns a dense representation of PauliOp.

Returns
-------
np.ndarray
    2D numpy array of complex numbers with a shape of :math:`2^n \times 2^n, n` - number of qubits
    )%")
        .def(
            "clone", [](fp::PauliOp<float_type> const &self) { return fp::PauliOp<float_type>(self); },
            R"%(Returns a copy of the PauliOp object.

Returns
-------
PauliOp
    A copy of the PauliOp object
)%")

        .def("__getstate__",
             [](fp::PauliOp<float_type> const &self) { return std::make_tuple(self.coeffs, self.pauli_strings); })
        .def("__setstate__",
             [](fp::PauliOp<float_type> &self, std::tuple<std::vector<cfloat_t>, std::vector<fp::PauliString>> state) {
                 new (&self) fp::PauliOp<float_type>(std::get<0>(state), std::get<1>(state));
             });

    ;
}

#endif
