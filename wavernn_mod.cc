#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>>
#include <vector>

#include "gru.h"
namespace py = pybind11;

using fvec = std::vector<float>;
using ivec = std::vector<int>;
using fndarray = py::array_t<float>;
using indarray = py::array_t<int>;

struct WaveRNN {
  int input_dim, hidden_dim;
  GRU gru;
  WaveRNN(int input_dim, int hidden_dim)
      : input_dim(input_dim),
        hidden_dim(hidden_dim),
        gru(input_dim, hidden_dim) {}

  void load_embed(fndarray embed) {
    auto a_embed = embed.unchecked<2>();
    for (int i = 0; i < 256; i++) {
      for (int j = 0; j < input_dim; j++) gru.embed[i][j] = a_embed(i, j);
    }
  }

  mat load_linear(vec& bias, fndarray w, indarray mask, fndarray b) {
    auto w_ptr = static_cast<float*>(w.request().ptr);
    auto mask_ptr = static_cast<int*>(mask.request().ptr);
    auto rb = b.unchecked<1>();
    // load bias
    for (int i = 0; i < rb.shape(0); i++) bias[i] = rb(i);
    // load weights
    masked_mat mm(w.shape(0), w.shape(1), mask_ptr, w_ptr);
    std::string buffer;
    mat mmm(mm);
    return mmm;
  }

  void load_weights(fndarray m1, indarray m1_mask, fndarray b1, fndarray m2,
                    indarray m2_mask, fndarray b2, fndarray m3,
                    indarray m3_mask, fndarray b3, fndarray o1,
                    indarray o1_mask, fndarray o1b, fndarray o2,
                    indarray o2_mask, fndarray o2b) {
    gru.m1 = load_linear(gru.b1, m1, m1_mask, b1);
    gru.m2 = load_linear(gru.b2, m1, m1_mask, b2);
    gru.m3 = load_linear(gru.b3, m1, m1_mask, b3);
    gru.o1 = load_linear(gru.o1b, o1, o1_mask, o1b);
    gru.o2 = load_linear(gru.o2b, o2, o2_mask, o2b);
  }

  std::vector<int> inference(fndarray ft) {
    auto rft = ft.unchecked<2>();
    std::vector<vec> xs;
    for (int i = 0; i < rft.shape(0); i++) {
      xs.emplace_back(input_dim);
      for (int j = 0; j < input_dim; j++) xs[i][j] = rft(i, j);
    }
    auto signal = gru.forward(xs);
    return signal;
  }
};

PYBIND11_MODULE(wavernn_mod, m) {
  py::class_<WaveRNN>(m, "WaveRNN")
      .def(py::init<int, int>())
      .def("load_embed", &WaveRNN::load_embed)
      .def("load_weights", &WaveRNN::load_weights)
      .def("inference", &WaveRNN::inference);
}
