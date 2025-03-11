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

void init_paulistring_bindings(nb::module_ &m)
{
    // TODO init default threading behavior for the module
    // TODO give up GIL when calling into long-running C++ code
    // TODO != and == operators for our Pauli structures
    using float_type = double;
    using cfloat_t = std::complex<float_type>;

    nb::class_<fp::PauliString>(
        m, "PauliString",
        R"%(A class representation of a Pauli String :math:`\mathcal{\hat{P}}` (i.e. a tensor product of Pauli matrices)

.. math::
    \mathcal{\hat{P}} = \bigotimes_i \sigma_i

    \sigma_i \in \{ I,X,Y,Z \}
)%")
        // Constructors
        .def(nb::init<>(), "Default constructor to initialize with empty string.")
        .def(
            nb::init<std::string const &>(), "string"_a,
            R"%(Constructs a PauliString from a string and calculates the weight. This is often the most compact way to initialize a PauliString.

Parameters
----------
string : str
    Pauli String representation. Each character should be one of :math:`I, X, Y, Z`
)%")
        .def(nb::init<std::vector<fp::Pauli> &>(), "paulis"_a,
             R"%(Constructs a PauliString from a list of Pauli objects and calculates the weight.

Parameters
----------
paulis : list[Pauli]
    List of ordered Pauli objects
)%")
        .def(
            "__str__", [](fp::PauliString const &self) { return fmt::format("{}", self); },
            R"%(Returns a string representation of PauliString object.

Returns
-------
str
    string representation of PauliString object
)%")
        .def(
            "__matmul__", [](fp::PauliString const &self, fp::PauliString const &rhs) { return self * rhs; },
            nb::is_operator(),
            R"%(Returns matrix product of two pauli strings and their phase as a pair.

Parameters
----------
rhs : PauliString
    Right hand side PauliString object

Returns
-------
tuple[complex, PauliString]
    Phase and resulting PauliString object
)%")
        .def(
            "__add__",
            [](fp::PauliString const &self, fp::PauliString const &other) {
                return fp::PauliOp<float_type>({self, other});
            },
            nb::is_operator(),
            R"%(Returns the sum of two Pauli strings in a form of PauliOp object.

Parameters
----------
rhs : PauliString
    The other PauliString object to add

Returns
-------
PauliOp
    A linear combination of the PauliString objects as a PauliOp.
)%")
        .def(
            "__sub__",
            [](fp::PauliString const &self, fp::PauliString const &other) {
                return fp::PauliOp<float_type>({1, -1}, {self, other});
            },
            nb::is_operator(),
            R"%(Returns the difference of two Pauli strings in a form of PauliOp object.

Parameters
----------
rhs : PauliString
    The other PauliString object to subtract

Returns
-------
PauliOp
    A linear combination of the PauliString objects as a PauliOp.
)%")

        // Properties
        .def_prop_ro("n_qubits", &fp::PauliString::n_qubits,
                     "int: The number of qubits in PauliString (i.e. number of Pauli Matrices in tensor product)")
        .def_prop_ro("dim", &fp::PauliString::dim,
                     "int: The dimension of PauliString :math:`2^n, n` - number of qubits")
        .def_prop_ro(
            "weight", [](fp::PauliString const &self) { return self.weight; },
            "int: The weight of PauliString (i.e. number of non-identity Pauli matrices in it)")
        // Methods
        .def(
            "apply",
            [](fp::PauliString const &self, nb::ndarray<cfloat_t> states, cfloat_t c) {
                // TODO handle the non-transposed case since that's likely the most common
                // TODO we should handle when users pass a single state (i.e. a 1D array here)
                if (states.ndim() == 1)
                {
                    // TODO lots of duplicate code here
                    auto states_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 1>(states);
                    auto new_states = fp::__detail::owning_ndarray_like_mdspan<cfloat_t, 1>(states_mdspan);
                    auto new_states_mdspan = std::mdspan(new_states.data(), new_states.size());
                    self.apply(std::execution::par, new_states_mdspan, states_mdspan);
                    return new_states;
                }
                else if (states.ndim() == 2)
                {
                    auto states_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 2>(states);
                    auto new_states = fp::__detail::owning_ndarray_like_mdspan<cfloat_t, 2>(states_mdspan);
                    auto new_states_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 2>(new_states);
                    self.apply_batch(std::execution::par, new_states_mdspan, states_mdspan, c);
                    return new_states;
                }
                else
                {
                    throw std::invalid_argument(
                        fmt::format("apply: expected 1 or 2 dimensions, got {}", states.ndim()));
                }
            },
            "states"_a, "coeff"_a = cfloat_t{1.0},
            R"%(Apply a Pauli string to a single dimensional state vector or a batch of states.

.. math::
    c \mathcal{\hat{P}} \ket{\psi_t}

.. note::
    For batch mode it applies the PauliString to each individual state separately.
    In this case, the input array is expected to have the shape of (n_dims, n_states) with states stored as columns.

Parameters
----------
states : np.ndarray
    The original state(s) represented as 1D (n_dims,) or 2D numpy array (n_dims, n_states) for batched calculation.
    Outer dimension must match the dimensionality of Pauli string.
coeff : complex
    Scalar multiplication factor (:math:`c`) to scale the PauliString before applying to states

Returns
-------
np.ndarray
    New state(s) in a form of 1D (n_dims,) or 2D numpy array (n_dims, n_states) according to the shape of input states
)%")
        .def(
            "expectation_value",
            [](fp::PauliString const &self, nb::ndarray<cfloat_t> states, cfloat_t c) {
                if (states.ndim() == 1)
                {
                    auto states_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 1>(states);
                    auto states_mdspan_2d = std::mdspan(states_mdspan.data_handle(), states_mdspan.extent(0), 1);
                    std::array<size_t, 1> out_shape = {1};
                    auto expected_vals_out = fp::__detail::owning_ndarray_from_shape<cfloat_t, 1>(out_shape);
                    auto expected_vals_out_mdspan = std::mdspan(expected_vals_out.data(), 1);
                    self.expectation_value(std::execution::par, expected_vals_out_mdspan, states_mdspan_2d, c);

                    return expected_vals_out;
                }
                else if (states.ndim() == 2)
                {
                    auto states_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 2>(states);
                    std::array<size_t, 1> out_shape = {states_mdspan.extent(1)};
                    auto expected_vals_out = fp::__detail::owning_ndarray_from_shape<cfloat_t, 1>(out_shape);
                    auto expected_vals_out_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 1>(expected_vals_out);
                    self.expectation_value(std::execution::par, expected_vals_out_mdspan, states_mdspan, c);
                    return expected_vals_out;
                }
                else
                {
                    throw std::invalid_argument(
                        fmt::format("expectation_value: expected 1 or 2 dimensions, got {}", states.ndim()));
                }
            },
            "states"_a, "coeff"_a = cfloat_t{1.0},
            R"%(Calculate expectation value(s) for a given single dimensional state vector or a batch of states.

.. math::
    \bra{\psi_t} \mathcal{\hat{P}} \ket{\psi_t}

.. note::
    For batch mode it computes the expectation value for each individual state separately.
    In this case, the input array is expected to have the shape of (n_dims, n_states) with states stored as columns.

Parameters
----------
states : np.ndarray
    The original state(s) represented as 1D (n_dims,) or 2D numpy array (n_dims, n_states) for batched calculation.
    Outer dimension must match the dimensionality of Pauli string.
coeff : complex
    Multiplication factor to scale the PauliString before calculating the expectation value

Returns
-------
np.ndarray
    Expectation value(s) in the form of a 1D numpy array with a shape of (n_states,)
)%")
        .def(
            "to_tensor",
            [](fp::PauliString const &self) {
                auto dense_pauli_str = fp::__detail::owning_ndarray_from_shape<cfloat_t, 2>({self.dim(), self.dim()});
                self.to_tensor(fp::__detail::ndarray_to_mdspan<cfloat_t, 2>(dense_pauli_str));
                return dense_pauli_str;
            },
            R"%(Returns a dense representation of PauliString.

Returns
-------
np.ndarray
    2D numpy array of complex numbers
    )%")
        .def(
            "clone", [](fp::PauliString const &self) { return fp::PauliString(self); },
            R"%(Returns a copy of the PauliString object.

Returns
-------
PauliString
    A copy of the PauliString object
)%")
        .def("__getstate__", [](fp::PauliString const &self) { return self.paulis; })
        .def("__setstate__", [](fp::PauliString &self, std::vector<fp::Pauli> paulis) {
            new (&self) fp::PauliString(std::move(paulis));
        });
}