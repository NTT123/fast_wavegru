/*
Test a simple GRU
*/

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <vector>

namespace py = pybind11;

using fvec = std::vector<float>;
using fndarray = py::array_t<float>;

#include <iostream>
#include <random>
#include <vector>

#include "sparse_matmul/sparse_matmul.h"

using namespace std;

using fvec = std::vector<float>;
using ivec = std::vector<int16_t>;
using fndarray = py::array_t<float>;
using indarray = py::array_t<int16_t>;

using mat = csrblocksparse::CsrBlockSparseMatrix<float, float, int16_t>;
using vec = csrblocksparse::CacheAlignedVector<float>;
using masked_mat = csrblocksparse::MaskedSparseMatrix<float>;

mat create_mat(int h, int w) {
  auto m = masked_mat(w, h, 0.90, 4, 4, 0.0, true);
  auto a = mat(m);
  return a;
}

struct M1 {
  mat m1;
  vec b1, z;
  int input_dim;
  int hidden_dim;
  M1(int input_dim, int hidden_dim)
      : input_dim(input_dim),
        hidden_dim(hidden_dim),
        b1(input_dim),
        z(hidden_dim) {
    m1 = create_mat(input_dim, hidden_dim);
  }

  mat load_linear(vec& bias, fndarray w, indarray mask, fndarray b) {
    auto w_ptr = static_cast<float*>(w.request().ptr);
    auto mask_ptr = static_cast<int16_t*>(mask.request().ptr);
    auto rb = b.unchecked<1>();
    // load bias
    for (int i = 0; i < rb.shape(0); i++) bias[i] = rb(i)/4;
    // load weights
    // bias.Print();
    masked_mat mm(w.shape(0), w.shape(1), mask_ptr, w_ptr);
    // mm.Print();
    mat mmm(mm);
    // mmm.Print();

    return mmm;
  }

  void load_weights(fndarray mm1, indarray m1_mask, fndarray bb1) {
    m1 = load_linear(b1, mm1, m1_mask, bb1);
  }

  void forward(fndarray x) {
    vec t(input_dim);
    auto rx = x.unchecked<1>();
    for (int i = 0; i < input_dim; i++) t[i] = rx(i);
    z.FillZero();
    m1.SpMM_bias(t, b1, &z, false);
    z.Print();
  }
};

PYBIND11_MODULE(m1_mod, m) {
  py::class_<M1>(m, "M1")
      .def(py::init<int, int>())
      .def("load_weights", &M1::load_weights)
      .def("forward", &M1::forward);
}
