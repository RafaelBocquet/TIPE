#pragma once

#include "Matrix.h"

#include <queue>
#include <iostream>
#include <random>

class Rna {
public:
  Rna(std::vector<unsigned> const& layer_sizes) : 
  random_generator(195486732){
    assert(layer_sizes.back() == 2);
    in_size = layer_sizes[0];
    for(unsigned i = 0; i < layer_sizes.size() - 1; ++i){
      layers.emplace_back(layer_sizes[i], layer_sizes[i + 1]);
    }
    std::normal_distribution<float> distribution(0.0, 1.0);
    for(Matrix<float>& mat : layers){
      mat.init([&](int, int) -> float {
        return distribution(random_generator);
      });
    }
  }

  std::uint16_t prediction(std::uint64_t bitpos, bool nxt) {
    return 1 << 15;
    nxt = nxt != 0;

    auto activation_function = [](float x) -> float { return 1.0 / (1.0 + exp(-x)); };
    auto activation_derivative = [&](float x) -> float { return activation_function(x) * (1 - activation_function(x)); };
    float training_rate = 0.05;

    // --- Hash context ---
    Matrix<float> in_vec(1, in_size);
    for(unsigned i = 0; i < in_size; ++i){ in_vec.at(0, i) = 0.0; }
    in_vec.at(0, bitpos % 8) = 1.0;
    in_vec.at(0, (0 + 8) % in_size) = 1.0;
    unsigned ctxidx = 0;
    for(unsigned i = 0; i < bit_queue.size(); ++i){
      ctxidx += (1 << i) * bit_queue[i];
      in_vec.at(0, (8 + ((1 << (i + 1)) - 1 + ctxidx)) % in_size) = 1.0;
    }
    // --- Create except vector ---
    Matrix<float> except(1, 1, [&](unsigned, unsigned j) -> float { return (nxt == j) ? 0.99 : 0.01; });
    // std::cout << "Except : " << except.at(0, 0) << std::endl;
    // --- Calc output ---
    std::vector<std::pair<Matrix<float>, Matrix<float>>> calculate;
    calculate.reserve(layers.size() + 1);
    calculate.emplace_back(in_vec, in_vec);
    for(Matrix<float>& mat : layers){
      in_vec = mat * in_vec;
      calculate.emplace_back(in_vec, in_vec);
      calculate.back().second.apply([&](unsigned, unsigned, float val) -> float {
        return activation_function(val);
      });
      calculate.back().first.apply([&](unsigned, unsigned, float val) -> float {
        return activation_derivative(val);
      });
      in_vec = calculate.back().first;
    }

    // --- Compute prediction ---
    float predic = in_vec.at(0, 0) / (in_vec.at(0, 0) + in_vec.at(0, 1));
    // std::cout << "Get : " << predic << std::endl;

    // --- Train network ---
    // --- Calculate deltas ---
    std::vector<std::vector<float>> delta;
    delta.resize(layers.size());
    for(int i = 0; i < layers.size(); ++i){
      delta[i].resize(layers[i].height());
    }
    // --- First delta ---
    #pragma omp parallel for
    for(int i = 0; i < layers.rbegin()->height(); ++i){
      delta[layers.size() - 1][i] =
        (calculate[layers.size()].second.at(0, i) - except.at(0, i)) * calculate[layers.size()].first.at(0, i);
    }
    // --- Backpropagation deltas ---
    for(int l = layers.size() - 2; l >= 0; --l){
      #pragma omp parallel for
      for(int i = 0; i < layers[l].height(); ++i){
        delta[l][i] = 0;
        #pragma omp parallel for
        for(int n = 0; n < layers[l+1].height(); ++n){
          delta[l][i] += layers[l+1].at(n, i) * delta[l+1][n];
        }
        delta[l][i] *= calculate[l+1].first.at(0, i);
      }
    }
    // --- Apply computation ---
    #pragma omp parallel for
    for(int l = 0; l < layers.size(); ++l){
      #pragma omp parallel for
      for(int i = 0; i < layers[l].height(); ++i){
        #pragma omp parallel for
        for(int j = 0; j < layers[l].width(); ++j){
          layers[l].at(i, j) -= training_rate * delta[l][i] * calculate[l].second.at(0, j);
        }
      }
    }

    // --- Manage context ---
    bit_queue.push_front(nxt);
    if(bit_queue.size() > 16){ bit_queue.pop_back(); }

    return static_cast<std::uint16_t>(65536.0 * predic);
  }

private:
  unsigned in_size;
  std::vector<Matrix<float>> layers;
  std::deque<bool> bit_queue;
  std::default_random_engine random_generator;
};