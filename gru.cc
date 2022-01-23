/*
Test a simple GRU
*/

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "sparse_matmul/sparse_matmul.h"

using namespace std;

using mat = csrblocksparse::CsrBlockSparseMatrix<float, float, int16_t>;
using vec = csrblocksparse::CacheAlignedVector<float>;
using masked_mat = csrblocksparse::MaskedSparseMatrix<float>;
std::bernoulli_distribution distribution(0.05);
std::default_random_engine generator;

mat create_mat(int h, int w) {
  auto m = masked_mat(w, h, 0.95, 4, 4, 0.0, true);
  auto a =mat(m);
  return a ;
}

struct GRU {
  int input_dim;
  int hidden_dim;
  vec b1, b2, b3;
  mat m11, m12, m21, m22, m31, m32;
  vec o11, o12, o21, o22, o31, o32, o1, o2, o3;
  vec hr;

  GRU(int input_dim, int hidden_dim)
      : input_dim(input_dim), hidden_dim(hidden_dim) {
    b1 = vec(hidden_dim);
    b1.FillRandom();
    b2 = vec(hidden_dim);
    b2.FillRandom();
    b3 = vec(hidden_dim);
    b3.FillRandom();
    o11 = vec(hidden_dim);
    o11.FillZero();
    o12 = vec(hidden_dim);
    o12.FillZero();
    o21 = vec(hidden_dim);
    o21.FillZero();
    o22 = vec(hidden_dim);
    o22.FillZero();
    o31 = vec(hidden_dim);
    o31.FillZero();
    o32 = vec(hidden_dim);
    o32.FillZero();

    o1 = vec(hidden_dim);
    o1.FillZero();
    o2 = vec(hidden_dim);
    o2.FillZero();
    o3 = vec(hidden_dim);
    o3.FillZero();
    hr = vec(hidden_dim);

    m11 = create_mat(input_dim, hidden_dim);
    m12 = create_mat(hidden_dim, hidden_dim);

    m21 = create_mat(input_dim, hidden_dim);
    m22 = create_mat(hidden_dim, hidden_dim);

    m31 = create_mat(input_dim, hidden_dim);
    m32 = create_mat(hidden_dim, hidden_dim);
  }

  void forward(vec& h, vec x) {
    m11.SpMM_bias(x, b1, &o11, false);
    m12.SpMM_bias(h, b1, &o12, false);
    csrblocksparse::detail::SumVectors(0, hidden_dim, o11.data(), o12.data(),
                                       o1.data());
    m21.SpMM_bias(x, b2, &o21, false);
    m22.SpMM_bias(h, b2, &o22, false);
    csrblocksparse::detail::SumVectors(0, hidden_dim, o21.data(), o22.data(),
                                       o2.data());

    o1.Sigmoid();
    o2.Sigmoid();

    for (int i = 0; i < hidden_dim; i++) {
      hr[i] = h[i] * o2[i];
    }

    m31.SpMM_bias(x, b3, &o31, false);
    m32.SpMM_bias(hr, b3, &o32, false);
    csrblocksparse::detail::SumVectors(0, hidden_dim, o31.data(), o32.data(),
                                       o3.data());
    o3.Tanh();
    for (int i = 0; i < hidden_dim; i++) {
      h[i] = (1. - o1[i]) * h[i] + o1[i] * o3[i];
    }
  }
};

int main() {
  GRU rnn(512, 1024);
  cout <<  rnn.m11.block_height() << endl;
  cout <<  rnn.m11.block_width() << endl;
  vec h(1024);
  h.FillRandom();
  vec x(512);
  x.FillRandom();
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 16000; i++) {
    rnn.forward(h, x); 
  }
  auto stop = std::chrono::high_resolution_clock::now();
  // h.Print();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  cout << duration.count() / 1e6 << endl;
  return 0;
}