#include "fast_pauli.hpp"
#include "__nb_helpers.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <vector>

namespace fp = fast_pauli;

/*
Python Bindings for PauliOp
*/

NB_MODULE(_fast_pauli, m) {
  // TODO init default threading behavior for the module
  // TODO give up GIL when calling into long-running C++ code
  using float_type = double;
  using cfloat_t = std::complex<float_type>;

  nb::class_<fp::Pauli>(m, "Pauli")
      // Constructors
      .def(nb::init<>())
      .def(nb::init<int const>(), "code"_a)
      .def(nb::init<char const>(), "symbol"_a)

      // Methods
      .def(
          "__mult__",
          [](fp::Pauli const &self, fp::Pauli const &rhs) {
            return self * rhs;
          },
          nb::is_operator())
      // TODO have this return numpy
      .def("to_tensor", &fp::Pauli::to_tensor<float_type>)
      .def("multiply", [](fp::Pauli const &self,
                          fp::Pauli const &rhs) { return self * rhs; })
      .def("__str__",
           [](fp::Pauli const &self) { return fmt::format("{}", self); });

  //
  //
  //

  nb::class_<fp::PauliString>(m, "PauliString")
      // Constructors
      .def(nb::init<>())
      .def(nb::init<std::string const &>(), "string"_a)
      .def(nb::init<std::vector<fp::Pauli> &>(), "paulis"_a)

      //
      .def("__str__",
           [](fp::PauliString const &self) { return fmt::format("{}", self); })

      // Properties
      .def_prop_ro("n_qubits", &fp::PauliString::n_qubits)
      .def_prop_ro("dim", &fp::PauliString::dim)
      .def_prop_ro("weight",
                   [](fp::PauliString const &self) { return self.weight; })
      // Methods
      .def(
          "apply",
          [](fp::PauliString const &self, nb::ndarray<cfloat_t> states,
             cfloat_t c) {
            // TODO handle the non-transposed case since that's likely the most
            // common

            if (states.ndim() == 1) {
              // TODO lots of duplicate code here
              // clang-format off
               auto states_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 1>(states);
               auto new_states = fp::__detail::owning_ndarray_like_mdspan<cfloat_t, 1>(states_mdspan);
               auto new_states_mdspan = std::mdspan(new_states.data(), new_states.size());
               self.apply(new_states_mdspan, states_mdspan);
              // clang-format on
              return new_states;

            } else if (states.ndim() == 2) {
              // clang-format off
               auto states_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 2>(states);
               auto new_states = fp::__detail::owning_ndarray_like_mdspan<cfloat_t, 2>(states_mdspan);
               auto new_states_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 2>(new_states);
               self.apply_batch(new_states_mdspan, states_mdspan, c);
              // clang-format on
              return new_states;
            } else {
              throw std::invalid_argument(fmt::format(
                  "apply: expected 1 or 2 dimensions, got {}", states.ndim()));
            }
          },
          "states"_a, "coeff"_a = cfloat_t{1.0})
      .def(
          // TODO we should handle when users pass a single state (i.e. a 1D
          // array here)
          "expectation_value",
          [](fp::PauliString const &self, nb::ndarray<cfloat_t> states,
             cfloat_t c) {
            if (states.ndim() == 1) {
              // clang-format off
              auto states_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 1>(states);
              auto states_mdspan_2d = std::mdspan(states_mdspan.data_handle(), states_mdspan.extent(0), 1);
              std::array<size_t, 1> out_shape = {1};
              auto expected_vals_out = fp::__detail::owning_ndarray_from_shape<cfloat_t, 1>(out_shape);
              auto expected_vals_out_mdspan = std::mdspan(expected_vals_out.data(), 1);
              self.expectation_value(expected_vals_out_mdspan, states_mdspan_2d, c);
              // clang-format on

              return expected_vals_out;
            } else if (states.ndim() == 2) {
              // clang-format off
              auto states_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 2>(states);
              std::array<size_t, 1> out_shape = {states_mdspan.extent(1)};
              auto expected_vals_out = fp::__detail::owning_ndarray_from_shape<cfloat_t, 1>(out_shape);
              auto expected_vals_out_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 1>(expected_vals_out);
              self.expectation_value(expected_vals_out_mdspan, states_mdspan, c);
              // clang-format on
              return expected_vals_out;
            } else {
              throw std::invalid_argument(fmt::format(
                  "expectation_value: expected 1 or 2 dimensions, got {}",
                  states.ndim()));
            }
          },
          "states"_a, "coeff"_a = cfloat_t{1.0})
      // TODO return numpy array
      .def("to_tensor",
           [](fp::PauliString const &self) {
             return self.get_dense_repr<float_type>();
           })

      //
      ;

  //
  nb::class_<fp::PauliOp<float_type>>(m, "PauliOp")
      // Constructors
      .def(nb::init<>())
      .def(nb::init<std::vector<std::string> const &>(), "pauli_strings"_a)
      .def(nb::init<std::vector<fp::PauliString>>())
      .def("__init__",
           [](fp::PauliOp<float_type> *new_obj, nb::ndarray<cfloat_t> coeffs,
              std::vector<fp::PauliString> const &pauli_strings) {
             auto [coeffs_vec, _] =
                 fp::__detail::ndarray_to_raw<cfloat_t, 1>(coeffs);
             new (new_obj) fp::PauliOp<float_type>(coeffs_vec, pauli_strings);
           })
      .def("__init__",
           [](fp::PauliOp<float_type> *new_obj,
              std::vector<cfloat_t> coeffs_vec,
              std::vector<fp::PauliString> const &pauli_strings) {
             new (new_obj) fp::PauliOp<float_type>(coeffs_vec, pauli_strings);
           })
      .def("__init__",
           [](fp::PauliOp<float_type> *new_obj,
              std::vector<cfloat_t> coeffs_vec,
              std::vector<std::string> const &strings) {
             std::vector<fp::PauliString> pauli_strings;
             std::transform(strings.begin(), strings.end(),
                            std::back_inserter(pauli_strings),
                            [](std::string const &pauli) {
                              return fp::PauliString(pauli);
                            });
             new (new_obj) fp::PauliOp<float_type>(coeffs_vec, pauli_strings);
           })

      // Getters
      .def_prop_ro("dim", &fp::PauliOp<float_type>::dim)
      .def_prop_ro("n_qubits", &fp::PauliOp<float_type>::n_qubits)
      .def_prop_ro("n_pauli_strings", &fp::PauliOp<float_type>::n_pauli_strings)
      // TODO these may dangerous, keep an eye on them if users start modifying
      // internals
      .def_prop_ro(
          "coeffs",
          [](fp::PauliOp<float_type> const &self) { return self.coeffs; })
      .def_prop_ro("pauli_strings",
                   [](fp::PauliOp<float_type> const &self) {
                     return self.pauli_strings;
                   })
      .def_prop_ro("pauli_strings_as_str",
                   [](fp::PauliOp<float_type> const &self) {
                     //  return self.pauli_strings;
                     std::vector<std::string> strings(self.n_pauli_strings());
                     std::transform(self.pauli_strings.begin(),
                                    self.pauli_strings.end(), strings.begin(),
                                    [](fp::PauliString const &ps) {
                                      return fmt::format("{}", ps);
                                    });
                     return strings;
                   })

      // Methods
      .def("apply",
           [](fp::PauliOp<float_type> const &self,
              nb::ndarray<cfloat_t> states) {
             if (states.ndim() == 1) {
               auto states_mdspan =
                   fp::__detail::ndarray_to_mdspan<cfloat_t, 1>(states);
               auto states_mdspan_2d = std::mdspan(states_mdspan.data_handle(),
                                                   states_mdspan.extent(0), 1);
               auto new_states =
                   fp::__detail ::owning_ndarray_like_mdspan<cfloat_t, 1>(
                       states_mdspan);
               std::mdspan new_states_mdspan =
                   std::mdspan(new_states.data(), new_states.size(), 1);
               //  auto new_states_mdspan =
               //      fp::__detail::ndarray_to_mdspan<cfloat_t, 2>(new_states);
               self.apply(new_states_mdspan, states_mdspan_2d);
               return new_states;
             } else if (states.ndim() == 2) {
               auto states_mdspan =
                   fp::__detail::ndarray_to_mdspan<cfloat_t, 2>(states);
               auto new_states =
                   fp::__detail ::owning_ndarray_like_mdspan<cfloat_t, 2>(
                       states_mdspan);
               auto new_states_mdspan =
                   fp::__detail::ndarray_to_mdspan<cfloat_t, 2>(new_states);

               self.apply(new_states_mdspan, states_mdspan);

               return new_states;
             } else {
               throw std::invalid_argument(fmt::format(
                   "apply: expected 1 or 2 dimensions, got {}", states.ndim()));
             }
           })
      .def("expectation_value",
           [](fp::PauliOp<float_type> const &self,
              nb::ndarray<cfloat_t> states) {
             if (states.ndim() == 1) {
               auto states_mdspan =
                   fp::__detail::ndarray_to_mdspan<cfloat_t, 1>(states);
               auto states_mdspan_2d = std::mdspan(states_mdspan.data_handle(),
                                                   states_mdspan.extent(0), 1);
               std::array<size_t, 1> out_shape = {1};
               auto expected_vals_out =
                   fp::__detail::owning_ndarray_from_shape<cfloat_t, 1>(
                       out_shape);
               auto expected_vals_out_mdspan =
                   fp::__detail::ndarray_to_mdspan<cfloat_t, 1>(
                       expected_vals_out);
               self.expectation_value(expected_vals_out_mdspan,
                                      states_mdspan_2d);
               return expected_vals_out;
             } else if (states.ndim() == 2) {
               auto states_mdspan =
                   fp::__detail::ndarray_to_mdspan<cfloat_t, 2>(states);
               std::array<size_t, 1> out_shape = {states_mdspan.extent(1)};
               auto expected_vals_out =
                   fp::__detail::owning_ndarray_from_shape<cfloat_t, 1>(
                       out_shape);
               auto expected_vals_out_mdspan =
                   fp::__detail::ndarray_to_mdspan<cfloat_t, 1>(
                       expected_vals_out);

               self.expectation_value(expected_vals_out_mdspan, states_mdspan);

               return expected_vals_out;
             } else {
               throw std::invalid_argument(fmt::format(
                   "expectation_value: expected 1 or 2 dimensions, got {}",
                   states.ndim()));
             }
           })
      .def("to_tensor",
           [](fp::PauliOp<float_type> const &self) {
             return self.get_dense_repr();
           })
      //
      ;

  //
  nb::class_<fp::SummedPauliOp<float_type>>(m, "SummedPauliOp")
      // Constructors
      // See
      // https://nanobind.readthedocs.io/en/latest/api_core.html#_CPPv4IDpEN8nanobind4initE
      .def(nb::init<>())
      .def("__init__",
           [](fp::SummedPauliOp<float_type> *new_obj,
              std::vector<std::string> &pauli_strings,
              nb::ndarray<cfloat_t> coeffs) {
             //
             auto coeffs_mdspan =
                 fp::__detail::ndarray_to_mdspan<cfloat_t, 2>(coeffs);

             new (new_obj)
                 fp::SummedPauliOp<float_type>(pauli_strings, coeffs_mdspan);
           })

      .def_prop_ro("dim", &fp::SummedPauliOp<float_type>::dim)
      .def_prop_ro("n_operators", &fp::SummedPauliOp<float_type>::n_operators)
      .def_prop_ro("n_pauli_strings",
                   &fp::SummedPauliOp<float_type>::n_pauli_strings)

      .def("apply",
           [](fp::SummedPauliOp<float_type> const &self,
              nb::ndarray<cfloat_t> states, nb::ndarray<float_type> data) {
             auto states_mdspan =
                 fp::__detail::ndarray_to_mdspan<cfloat_t, 2>(states);
             auto data_mdspan =
                 fp::__detail::ndarray_to_mdspan<float_type, 2>(data);

             // clang-format off
             auto new_states        = fp::__detail::owning_ndarray_like_mdspan<cfloat_t, 2>(states_mdspan);
             auto new_states_mdspan = fp::__detail::ndarray_to_mdspan<cfloat_t, 2>(new_states);
             // clang-format on

             self.apply_parallel<float_type>(new_states_mdspan, states_mdspan,
                                             data_mdspan);

             return new_states;
           })
      //
      ;

  //
  // Helpers
  //
  auto helpers_m = m.def_submodule("helpers");
  helpers_m.def("get_nontrivial_paulis", &fp::get_nontrivial_paulis,
                "weight"_a);
  helpers_m.def("calcutate_pauli_strings", &fp::calcutate_pauli_strings,
                "n_qubits"_a, "weight"_a);
  helpers_m.def("calculate_pauli_strings_max_weight",
                &fp::calculate_pauli_strings_max_weight, "n_qubits"_a,
                "weight"_a);
  helpers_m.def("pauli_string_sparse_repr", &fp::get_sparse_repr<float_type>,
                "paulis"_a);
}