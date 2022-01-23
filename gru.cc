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
  auto m = masked_mat(w, h, 0.95, 4, 4, 0.0, true);
  auto a = mat(m);
  return a;
}

struct GRU {
  int input_dim;
  int hidden_dim;
  mat m1, m2, m3;
  vec b1, b2, b3;
  vec o1, o2, o3, lo;
  vec t;
  vec lb;
  mat lw;
  std::vector<vec> embed;

  GRU(int input_dim, int hidden_dim)
      : input_dim(input_dim),
        hidden_dim(hidden_dim),
        b1(hidden_dim),
        b2(hidden_dim),
        b3(hidden_dim),
        o1(hidden_dim),
        o2(hidden_dim),
        o3(hidden_dim),
        lo(256),
        t(hidden_dim + input_dim),
        lb(256) {
    m1 = create_mat(input_dim + hidden_dim, hidden_dim);
    m2 = create_mat(input_dim + hidden_dim, hidden_dim);
    m3 = create_mat(input_dim + hidden_dim, hidden_dim);
    lw = create_mat(hidden_dim, 256);
    embed = std::vector<vec>();
    for (int i = 0; i < 256; i++) {
      embed.emplace_back(512);
      embed[i].FillRandom();
    }
  }

  void load_weights(std::vector<float> m1, std::vector<int> mask_m1,
                    std::vector<int> b1, std::vector<float> m2,
                    std::vector<int> mask_m2, std::vector<int> b2,
                    std::vector<float> m3, std::vector<int> mask_m3,
                    std::vector<int> b3) {}

  void forward(vec& h, std::vector<vec>& xs) {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    int value = 127;
    std::vector<int> signal(xs.size());
    std::cout << xs.size() << std::endl;
    for (int index = 0; index < xs.size(); index++) {
      for (int i = 0; i < input_dim; i++) t[i] = xs[index][i] + embed[value][i];
      for (int i = 0; i < hidden_dim; i++) t[input_dim + i] = h[i];
      m1.SpMM_bias(t, b1, &o1, false);
      m2.SpMM_bias(t, b2, &o2, false);
      o1.Sigmoid();
      o2.Sigmoid();

      for (int i = 0; i < hidden_dim; i++) {
        t[input_dim + i] = h[i] * o2[i];
      }

      m3.SpMM_bias(t, b3, &o3, false);
      o3.Tanh();
      for (int i = 0; i < hidden_dim; i++) {
        h[i] = (1. - o1[i]) * h[i] + o1[i] * o3[i];
      }
      lw.SpMM_bias(h, lb, &lo, false);
      value = lo.Sample();
    }
  }
};

int main() {
  GRU rnn(512, 1024);
  vec h(1024);
  h.FillRandom();
  std::vector<vec> xs;
  for (int i = 0; i < 16000; i++) {
    xs.emplace_back(512);
    xs[i].FillRandom();
  }

  auto start = std::chrono::high_resolution_clock::now();
  rnn.forward(h, xs);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << duration.count() / 1e6 << endl;
  return 0;
}