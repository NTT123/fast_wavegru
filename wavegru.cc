#include <iostream>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>
#include "sparse_matmul/sparse_matmul.h"



using namespace std; 

void mytest() {

  const int kRows = 8;
  const int kCols = 8;
  std::vector<int> mask = {
      1, 1, 1, 1, 0, 0, 0, 0, 
      1, 1, 1, 1, 0, 0, 0, 0,
      1, 1, 1, 1, 0, 0, 0, 0, 
      1, 1, 1, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 1, 1, 1, 
      0, 0, 0, 0, 1, 1, 1, 1, 
      0, 0, 0, 0, 1, 1, 1, 1, 
      0, 0, 0, 0, 1, 1, 1, 1
      };
  std::vector<float> values(kRows * kCols, 1.f);
  values[1] = 2.f;
  values[3] = 3.f;
  values[36] = -1.f;
  values[45] = -2.f;

  csrblocksparse::CacheAlignedVector<float> bias(kRows);
  csrblocksparse::CacheAlignedVector<float> rhs(kCols);
  csrblocksparse::CacheAlignedVector<float> out_ref(kRows);
  csrblocksparse::CacheAlignedVector<float> out_test(kRows);

  bias.FillZero();
  rhs.FillOnes();

  rhs.Print();

  csrblocksparse::MaskedSparseMatrix<float> matrix(kRows, kCols, mask.data(),
                                                   values.data());

  matrix.SpMM_bias(rhs, bias, &out_ref);

  csrblocksparse::CsrBlockSparseMatrix<csrblocksparse::bfloat16, float, int16_t>
      block_sparse_matrix(matrix);

  std::string buffer;
  std::size_t num_bytes = block_sparse_matrix.WriteToFlatBuffer(&buffer);

  csrblocksparse::CsrBlockSparseMatrix<csrblocksparse::bfloat16, float, int16_t>
      new_block_sparse_matrix(reinterpret_cast<const uint8_t*>(buffer.c_str()),
                              num_bytes);

  new_block_sparse_matrix.SpMM_bias(rhs, bias, &out_test);

  out_test.Print();


}
int main() {
    cout << "hello world" << endl;
    mytest();
    cout << "hello world, again" << endl;
    return 0;
}