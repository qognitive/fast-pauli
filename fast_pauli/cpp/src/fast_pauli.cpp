#include "fast_pauli.hpp"

#include <experimental/mdspan>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace fp = fast_pauli;
namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(_fast_pauli, m) {
  // TODO init default threading behaviour for the module
  // TODO give up GIL when calling into long-running C++ code

  // TODO add hierarchy with more submodules like
  // _fast_pauli.helpers, _fast_pauli._core to expose internal algos ?

  py::class_<fp::Pauli>(m, "Pauli")
      .def(py::init<>())
      .def(py::init<int const>(), "code"_a)
      .def(py::init<char const>(), "symbol"_a)
      .def("to_tensor", &fp::Pauli::to_tensor<double>)
      .def("__str__",
           [](fp::Pauli const &self) { return fmt::format("{}", self); })
      .def("__mul__", [](fp::Pauli const &self, fp::Pauli const &rhs) {
        return self * rhs;
      });

  py::class_<fp::PauliString>(m, "PauliString")
      .def(py::init<>())
      .def(py::init([](std::vector<fp::Pauli> paulis) {
             return fp::PauliString(paulis);
           }),
           "paulis"_a)
      .def(py::init<std::string const &>(), "string"_a)
      .def_property_readonly("n_qubits", &fp::PauliString::n_qubits)
      .def_property_readonly("dims", &fp::PauliString::dims)
      .def_readonly("weight", &fp::PauliString::weight)
      .def("to_tensor", &fp::PauliString::get_dense_repr<double>)
      .def(
          "apply",
          [](fp::PauliString const &self,
             std::vector<std::complex<double>> vec) { return self.apply(vec); },
          "state"_a)
      .def(
          "apply_batch",
          // TODO: this should be handled by proper adapters for mdspan
          [](fp::PauliString const &self,
             std::vector<std::vector<std::complex<double>>> inputs,
             std::complex<double> coef) {
            if (inputs.empty())
              return std::vector<std::vector<std::complex<double>>>{};
            // for now we expect row major inputs which have states as columns
            size_t const n_states = inputs.front().size();
            std::vector<std::complex<double>> flat_inputs;
            flat_inputs.reserve(inputs.size() * n_states);

            for (auto const &vec : inputs)
              if (vec.size() != n_states)
                throw std::invalid_argument("Bad shape of states array");
              else
                std::ranges::copy(vec.begin(), vec.end(),
                                  std::back_inserter(flat_inputs));

            std::vector<std::complex<double>> flat_outputs(flat_inputs.size(),
                                                           0);
            self.apply_batch(
                std::mdspan<std::complex<double>, std::dextents<size_t, 2>>{
                    flat_outputs.data(), inputs.size(), n_states},
                std::mdspan<std::complex<double>, std::dextents<size_t, 2>>{
                    flat_inputs.data(), inputs.size(), n_states},
                coef);

            // TODO arrange this ugly converters into utility functions at least
            std::vector<std::vector<std::complex<double>>> results(
                inputs.size());
            for (size_t i = 0; i < inputs.size(); i++) {
              auto it = flat_outputs.begin() + i * n_states;
              std::ranges::copy(it, it + n_states,
                                std::back_inserter(results[i]));
            }
            return results;
          },
          "states"_a, "coeff"_a = std::complex<double>{1.0})
      .def("__str__",
           [](fp::PauliString const &self) { return fmt::format("{}", self); });

  py::class_<fp::SummedPauliOp<double>>(m, "SummedPauliOp").def(py::init<>());
}
