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

  py::class_<fp::SummedPauliOp<double>>(m, "SummedPauliOp").def(py::init<>());
}