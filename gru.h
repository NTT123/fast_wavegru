/*
Test a simple GRU
*/

#include <iostream>
#include <random>
#include <vector>

#include "sparse_matmul/sparse_matmul.h"

using namespace std;

using mat = csrblocksparse::CsrBlockSparseMatrix<float, float, int16_t>;
using vec = csrblocksparse::CacheAlignedVector<float>;
using masked_mat = csrblocksparse::MaskedSparseMatrix<float>;

mat create_mat(int h, int w) {
  auto m = masked_mat(w, h, 0.90, 4, 4, 0.0, true);
  auto a = mat(m);
  return a;
}

struct GRU {
  int input_dim;
  int hidden_dim;
  mat m1, m2, m3;
  vec b1, b2, b3;
  vec z, r, hh;
  vec fco1, fco2;
  vec o1b, o2b;
  vec t;
  vec lb1, lb2;
  vec h;
  mat o1, o2;
  std::vector<vec> embed;

  GRU(int input_dim, int hidden_dim)
      : input_dim(input_dim),
        hidden_dim(hidden_dim),
        b1(hidden_dim),
        b2(hidden_dim),
        b3(hidden_dim),
        z(hidden_dim),
        r(hidden_dim),
        hh(hidden_dim),
        fco1(512),
        fco2(256),
        t(hidden_dim + input_dim),
        h(hidden_dim),
        o1b(512),
        o2b(256) {
    m1 = create_mat(input_dim + hidden_dim, hidden_dim);
    m2 = create_mat(input_dim + hidden_dim, hidden_dim);
    m3 = create_mat(input_dim + hidden_dim, hidden_dim);
    o1 = create_mat(hidden_dim, 512);
    o2 = create_mat(512, 256);
    embed = std::vector<vec>();
    for (int i = 0; i < 256; i++) {
      embed.emplace_back(512);
      embed[i].FillRandom();
    }
  }

  std::vector<int> forward(std::vector<vec>& xs, float temperature) {
    int value = 127;
    std::vector<int> signal(xs.size());
    h.FillZero();
    for (int index = 0; index < xs.size(); index++) {
      for (int i = 0; i < input_dim; i++) t[i] = xs[index][i] + embed[value][i];
      for (int i = 0; i < hidden_dim; i++) t[input_dim + i] = h[i];
      m1.SpMM_bias(t, b1, &z, false);
      m2.SpMM_bias(t, b2, &r, false);
      z.Sigmoid();
      r.Sigmoid();

      for (int i = 0; i < hidden_dim; i++) {
        t[input_dim + i] = h[i] * r[i];
      }

      m3.SpMM_bias(t, b3, &hh, false);
      hh.Tanh();
      for (int i = 0; i < hidden_dim; i++) {
        h[i] = (1. - z[i]) * h[i] + z[i] * hh[i];
      }
      o1.SpMM_bias(h, o1b, &fco1, false);
      o2.SpMM_bias(fco1, o2b, &fco2, false);
      value = fco2.Sample(temperature);
      signal[index] = value;
    }
    return signal;
  }
};
