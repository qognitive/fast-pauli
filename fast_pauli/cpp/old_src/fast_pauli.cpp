#include "fast_pauli.hpp"
#include "__pauli_string.hpp"

#include <experimental/mdspan>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace fp = fast_pauli;
namespace py = pybind11;
using namespace pybind11::literals;

namespace fast_pauli {

/**
 * @brief Flatten row major matrix represented as a 2D-std::vector into single
 * std::vector
 *
 * @tparam T The floating point base to use for all the complex numbers
 * @return  std::vector<std::complex<T>> flattened vector with rows concatenated
 */
template <std::floating_point T>
inline std::vector<std::complex<T>>
flatten_vector(std::vector<std::vector<std::complex<T>>> const &inputs) {
  auto const n_rows = inputs.size();
  auto const n_cols = inputs.front().size();
  std::vector<std::complex<T>> flat;
  flat.reserve(n_rows * n_cols);

  for (auto const &vec : inputs)
    if (vec.size() != n_cols)
      throw std::invalid_argument("Bad shape of input array");
    else
      std::ranges::copy(vec.begin(), vec.end(), std::back_inserter(flat));

  return flat;
}

} // namespace fast_pauli

PYBIND11_MODULE(_fast_pauli, m) {
  // TODO init default threading behaviour for the module
  // TODO give up GIL when calling into long-running C++ code
  using float_type = double;

  py::class_<fp::Pauli>(m, "Pauli")
      .def(py::init<>())
      .def(py::init<int const>(), "code"_a)
      .def(py::init<char const>(), "symbol"_a)
      .def("to_tensor", &fp::Pauli::to_tensor<float_type>)
      .def("multiply", [](fp::Pauli const &self,
                          fp::Pauli const &rhs) { return self * rhs; })
      .def("__str__",
           [](fp::Pauli const &self) { return fmt::format("{}", self); });

  // TODO should we have PauliString templated on the float_type as well,
  // instead of each individual method?
  py::class_<fp::PauliString>(m, "PauliString")
      .def(py::init<>())
      .def(py::init([](std::vector<fp::Pauli> paulis) {
             return fp::PauliString(paulis);
           }),
           "paulis"_a)
      .def(py::init<std::string const &>(), "string"_a)
      .def_property_readonly("n_qubits", &fp::PauliString::n_qubits)
      .def_property_readonly("dim", &fp::PauliString::dim)
      .def_readonly("weight", &fp::PauliString::weight)
      .def("to_tensor", &fp::PauliString::get_dense_repr<float_type>)
      .def(
          "apply",
          [](fp::PauliString const &self,
             std::vector<std::complex<float_type>> state) {
            return self.apply(state);
          },
          "state"_a)
      .def(
          "apply",
          // TODO: this should be handled by proper adapters for mdspan
          [](fp::PauliString const &self,
             std::vector<std::vector<std::complex<float_type>>> states,
             std::complex<float_type> coeff) {
            if (states.empty())
              return std::vector<std::vector<std::complex<float_type>>>{};
            // for now we expect row major inputs which have states as columns
            auto flat_inputs = fp::flatten_vector(states);
            auto const n_states = states.front().size();
            std::vector<std::complex<float_type>> flat_outputs(
                flat_inputs.size(), 0);
            self.apply_batch(
                std::mdspan<std::complex<float_type>, std::dextents<size_t, 2>>{
                    flat_outputs.data(), states.size(), n_states},
                std::mdspan<std::complex<float_type>, std::dextents<size_t, 2>>{
                    flat_inputs.data(), states.size(), n_states},
                coeff);

            std::vector<std::vector<std::complex<float_type>>> results(
                states.size());
            for (size_t i = 0; i < states.size(); i++) {
              auto it = flat_outputs.begin() + i * n_states;
              std::ranges::copy(it, it + n_states,
                                std::back_inserter(results[i]));
            }
            return results;
          },
          "states"_a, "coeff"_a = std::complex<float_type>{1.0})
      // .def(
      //     "expectation_value",
      //     [](fp::PauliString const &self,
      //        std::vector<std::complex<float_type>> state) {
      //       std::mdspan<std::complex<float_type>, std::dextents<size_t, 2>>
      //           span_state{state.data(), state.size(), 1};
      //       std::vector<std::complex<float_type>> output(1, 0);
      //       std::mdspan<std::complex<float_type>, std::dextents<size_t, 1>>
      //           span_output{output.data(), 1};

      //       self.expectation_value(span_output, span_state);
      //       return output.at(0);
      //     },
      //     "state"_a)
      // .def(
      //     "expectation_value",
      //     [](fp::PauliString const &self,
      //        std::vector<std::vector<std::complex<float_type>>> states) {
      //       if (states.empty())
      //         return std::vector<std::complex<float_type>>{};
      //       auto flat_states = fp::flatten_vector(states);
      //       std::mdspan<std::complex<float_type>, std::dextents<size_t, 2>>
      //           span_states{flat_states.data(), states.size(),
      //                       states.front().size()};
      //       std::vector<std::complex<float_type>>
      //       output(states.front().size(),
      //                                                    0);
      //       std::mdspan<std::complex<float_type>, std::dextents<size_t, 1>>
      //           span_output{output.data(), output.size()};

      //       self.expectation_value(span_output, span_states);
      //       return output;
      //     },
      // "states"_a)
      .def("__str__",
           [](fp::PauliString const &self) { return fmt::format("{}", self); });

  using pauli_op_type = fp::PauliOp<float_type>;
  py::class_<pauli_op_type>(m, "PauliOp")
      .def(py::init<>())
      .def(py::init<std::vector<std::complex<float_type>>,
                    std::vector<fp::PauliString>>(),
           "coefficients"_a, "strings"_a)
      .def(py::init<std::vector<fp::PauliString>>(), "strings"_a)
      .def(py::init([](std::vector<std::complex<float_type>> coefficients,
                       std::vector<std::string> paulis) {
             std::vector<fp::PauliString> pauli_strings;
             std::transform(paulis.begin(), paulis.end(),
                            std::back_inserter(pauli_strings),
                            [](std::string const &pauli) {
                              return fp::PauliString(pauli);
                            });
             return pauli_op_type(std::move(coefficients),
                                  std::move(pauli_strings));
           }),
           "coefficients"_a, "strings"_a)
      .def_property_readonly("n_strings", &pauli_op_type::n_pauli_strings)
      .def_property_readonly("n_qubits", &pauli_op_type::n_qubits)
      .def_property_readonly("dim", &pauli_op_type::dim)
      .def_property_readonly(
          "coeffs", [](pauli_op_type const &self) { return self.coeffs; })
      .def_property_readonly(
          "strings",
          [](pauli_op_type const &self) -> std::vector<std::string> {
            auto convert = [](auto const &ps) { return fmt::format("{}", ps); };
            auto strings_view =
                self.pauli_strings | std::views::transform(convert);
            return {strings_view.begin(), strings_view.end()};
          })
      .def("to_tensor", &pauli_op_type::get_dense_repr)
      // two bindings below is just barefaced copy-paste from PauliString
      .def(
          "apply",
          [](pauli_op_type const &self,
             std::vector<std::complex<float_type>> state) {
            return self.apply(state);
          },
          "state"_a)
      .def(
          "apply",
          [](pauli_op_type const &self,
             std::vector<std::vector<std::complex<float_type>>> states) {
            if (states.empty())
              return std::vector<std::vector<std::complex<float_type>>>{};
            // for now we expect row major inputs which have states as columns
            auto flat_inputs = fp::flatten_vector(states);
            auto const n_states = states.front().size();
            std::vector<std::complex<float_type>> flat_outputs(
                flat_inputs.size(), 0);
            self.apply(
                std::mdspan<std::complex<float_type>, std::dextents<size_t, 2>>{
                    flat_outputs.data(), states.size(), n_states},
                std::mdspan<std::complex<float_type>, std::dextents<size_t, 2>>{
                    flat_inputs.data(), states.size(), n_states});

            std::vector<std::vector<std::complex<float_type>>> results(
                states.size());
            for (size_t i = 0; i < states.size(); i++) {
              auto it = flat_outputs.begin() + i * n_states;
              std::ranges::copy(it, it + n_states,
                                std::back_inserter(results[i]));
            }
            return results;
          },
          "states"_a)
      // .def(
      //     "expectation_value",
      //     [](pauli_op_type const &self,
      //        std::vector<std::complex<float_type>> state) {
      //       std::mdspan<std::complex<float_type>, std::dextents<size_t, 2>>
      //           span_state{state.data(), state.size(), 1};
      //       auto output = self.expectation_value(span_state);
      //       return output.at(0);
      //     },
      //     "state"_a)
      // .def(
      //     "expectation_value",
      //     [](pauli_op_type const &self,
      //        std::vector<std::vector<std::complex<float_type>>> states) {
      //       if (states.empty())
      //         return std::vector<std::complex<float_type>>{};
      //       auto flat_states = fp::flatten_vector(states);
      //       std::mdspan<std::complex<float_type>, std::dextents<size_t, 2>>
      //           span_states{flat_states.data(), states.size(),
      //                       states.front().size()};
      //       return self.expectation_value(span_states);
      //     },
      //     "states"_a)
      ;

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
