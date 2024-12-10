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
Python Bindings for PauliOp
*/

NB_MODULE(_fast_pauli, m)
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

    //
    //
    //

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
)%");

    //
    //
    //

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
            "List[complex]: Ordered list of coefficients corresponding to Pauli strings")
        .def_prop_ro(
            "pauli_strings", [](fp::PauliOp<float_type> const &self) { return self.pauli_strings; },
            "List[PauliString]: Ordered list of PauliString objects in PauliOp")
        .def_prop_ro(
            "pauli_strings_as_str",
            [](fp::PauliOp<float_type> const &self) {
                //  return self.pauli_strings;
                std::vector<std::string> strings(self.n_pauli_strings());
                std::transform(self.pauli_strings.begin(), self.pauli_strings.end(), strings.begin(),
                               [](fp::PauliString const &ps) { return fmt::format("{}", ps); });
                return strings;
            },
            "List[str]: Ordered list of Pauli Strings representations from PauliOp")

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

        ;

    //
    //
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

        //
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
    The original state(s) represented as 2D numpy array (n_operators, n_states) for batched calculation.

Returns
-------
np.ndarray
    New state(s) in a form of 2D numpy array (n_operators, n_states) according to the shape of input states
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
    The original state(s) represented as 2D numpy array (n_operators, n_states) for batched calculation.
data : np.ndarray
    The data to weight the operators corresponding to the states (n_operators, n_states)

Returns
-------
np.ndarray
    New state(s) in a form of 2D numpy array (n_operators, n_states) according to the shape of input states
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
        .def("square", &fp::SummedPauliOp<float_type>::square, R"%(Square the SummedPauliOp.

Returns
-------
SummedPauliOp
    New SummedPauliOp instance
)%");
    //
    ;

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
)%");
}