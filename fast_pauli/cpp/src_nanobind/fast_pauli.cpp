#include "fast_pauli.hpp"
#include "__nb_helpers.hpp"
#include "nanobind/nanobind.h"

namespace fp = fast_pauli;

/*
Python Bindings for PauliOp
*/

NB_MODULE(fppy, m) {
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
      // TODO make names consistent
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
              // TODO we can do better with this, right now it takes a 1D array
              // and returns a 2D one which isn't very intuitive
              // clang-format off
               auto states_mdspan = ndarray_to_mdspan<cfloat_t, 1>(states);
               auto states_mdspan_2d =std::mdspan(states_mdspan.data_handle(),states_mdspan.extent(0),1);
               auto new_states = owning_ndarray_like_mdspan<cfloat_t, 2>(states_mdspan_2d);
               auto new_states_mdspan = ndarray_to_mdspan<cfloat_t, 2>(new_states);
               // TODO refactor PauliString::apply to match the apply_batch interface (i.e. no output and everything is an mdspan)
               self.apply_batch(new_states_mdspan, states_mdspan_2d, c);
              // clang-format on
              return new_states;

            } else if (states.ndim() == 2) {
              // clang-format off
               auto states_mdspan = ndarray_to_mdspan<cfloat_t, 2>(states);
               auto new_states = owning_ndarray_like_mdspan<cfloat_t, 2>(states_mdspan);
               auto new_states_mdspan = ndarray_to_mdspan<cfloat_t, 2>(new_states);
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
            auto states_mdspan = ndarray_to_mdspan<cfloat_t, 2>(states);
            std::array<size_t, 1> out_shape = {states_mdspan.extent(1)};
            auto expected_vals_out =
                owning_ndarray_from_shape<cfloat_t, 1>(out_shape);
            auto expected_vals_out_mdspan =
                ndarray_to_mdspan<cfloat_t, 1>(expected_vals_out);

            self.expectation_value(expected_vals_out_mdspan, states_mdspan, c);

            return expected_vals_out;
          },
          "states"_a, "coeff"_a = cfloat_t{1.0})
      // TODO return numpy array
      .def("get_dense_repr",
           [](fp::PauliString const &self) {
             return self.get_dense_repr<float_type>();
           })

      //
      ;

  //
  nb::class_<fp::PauliOp<float_type>>(m, "PauliOp")
      // Constructors
      .def(nb::init<>())
      .def(nb::init<std::vector<fp::PauliString>>())
      .def("__init__",
           [](fp::PauliOp<float_type> *new_obj,
              std::vector<fp::PauliString> const &pauli_strings,
              nb::ndarray<cfloat_t> coeffs) {
             auto [coeffs_vec, _] = ndarray_to_raw<cfloat_t, 1>(coeffs);
             new (new_obj) fp::PauliOp<float_type>(coeffs_vec, pauli_strings);
           })

      // Getters
      // TODO update the dims function name to n_dimenions
      .def_prop_ro("dimensions", &fp::PauliOp<float_type>::dim)
      .def_prop_ro("n_qubits", &fp::PauliOp<float_type>::n_qubits)
      // TODO update n_strings function name
      .def_prop_ro("n_pauli_strings", &fp::PauliOp<float_type>::n_pauli_strings)

      // Methods
      .def("apply",
           [](fp::PauliOp<float_type> const &self,
              nb::ndarray<cfloat_t> states) {
             auto states_mdspan = ndarray_to_mdspan<cfloat_t, 2>(states);
             auto new_states =
                 owning_ndarray_like_mdspan<cfloat_t, 2>(states_mdspan);
             auto new_states_mdspan =
                 ndarray_to_mdspan<cfloat_t, 2>(new_states);

             self.apply(new_states_mdspan, states_mdspan);

             return new_states;
           })
      // TODO update the expectation_value function to take mdspan
      // .def("expectation_value",
      //      [](fp::PauliOp<float_type> const &self,
      //         nb::ndarray<cfloat_t> states) {
      //        auto states_mdspan = ndarray_to_mdspan<cfloat_t, 2>(states);
      //        std::array<size_t, 1> out_shape = {states_mdspan.extent(1)};
      //        auto expected_vals_out =
      //            owning_ndarray_from_shape<cfloat_t, 1>(out_shape);
      //        auto expected_vals_out_mdspan =
      //            ndarray_to_mdspan<cfloat_t, 1>(expected_vals_out);

      //        self.expectation_value(expected_vals_out_mdspan, states_mdspan);

      //        return expected_vals_out;
      //      })
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
             auto coeffs_mdspan = ndarray_to_mdspan<cfloat_t, 2>(coeffs);

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
             auto states_mdspan = ndarray_to_mdspan<cfloat_t, 2>(states);
             auto data_mdspan = ndarray_to_mdspan<float_type, 2>(data);

             // clang-format off
             auto new_states        = owning_ndarray_like_mdspan<cfloat_t, 2>(states_mdspan);
             auto new_states_mdspan = ndarray_to_mdspan<cfloat_t, 2>(new_states);
             // clang-format on

             self.apply_parallel<float_type>(new_states_mdspan, states_mdspan,
                                             data_mdspan);

             return new_states;
           })
      //
      ;
}