#include "fast_pauli.hpp"

#include <experimental/mdspan>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace fp = fast_pauli;
namespace py = pybind11;
using namespace pybind11::literals;

void scale_tensor_3d(py::array_t<double> array, double scale) {
  auto arr = array.mutable_unchecked<>();
  std::mdspan tensor(arr.mutable_data(), arr.shape(0), arr.shape(1),
                     arr.shape(2));

#pragma omp parallel for collapse(3)
  for (size_t i = 0; i < tensor.extent(0); i++) {
    for (size_t j = 0; j < tensor.extent(1); j++) {
      for (size_t k = 0; k < tensor.extent(2); k++) {
        tensor(i, j, k) *= scale;
      }
    }
  }
}

PYBIND11_MODULE(_fast_pauli, m) {
  // TODO init default threading behaviour for the module

  m.doc() = "Example NumPy/C++ Interface Using std::mdspan"; // optional module
                                                             // docstring
  m.def("scale_tensor_3d", &scale_tensor_3d, "Scale a 3D tensor by a scalar.",
        py::arg().noconvert(), py::arg("scale"));

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
            // for now we expect row major inputs which have states as columns
            const size_t n_states = inputs[0].size();
            std::vector<std::complex<double>> flat_inputs;
            flat_inputs.reserve(inputs.size() * n_states);

            for (const auto &vec : inputs)
              if (vec.size() != n_states)
                throw std::invalid_argument("Bad shape of states array");
              else
                std::copy(vec.begin(), vec.end(),
                          std::back_inserter(flat_inputs));

            std::vector<std::complex<double>> flat_outputs(flat_inputs.size(),
                                                           0);
            self.apply_batch(
                std::mdspan<std::complex<double>, std::dextents<size_t, 2>>{
                    flat_outputs.data(), inputs.size(), n_states},
                std::mdspan<std::complex<double>, std::dextents<size_t, 2>>{
                    flat_inputs.data(), inputs.size(), n_states},
                coef);

            std::vector<std::vector<std::complex<double>>> results(
                inputs.size());
            for (size_t i = 0; i < inputs.size(); i++) {
              auto it = flat_outputs.begin() + i * n_states;
              std::copy(it, it + n_states, std::back_inserter(results[i]));
            }
            return results;
          },
          "states"_a, "coef"_a)
      .def("__str__",
           [](fp::PauliString const &self) { return fmt::format("{}", self); });

  py::class_<fp::SummedPauliOp<double>>(m, "SummedPauliOp").def(py::init<>());
}