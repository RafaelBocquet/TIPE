#pragma once
#include "Model.h"

// --- Rna model ---

template<unsigned CtxSize, unsigned ...LS>
class BitRNAModel : public Model {
  static constexpr unsigned LayerCount = 1 + sizeof...(LS);
  static constexpr unsigned InContextSize = (1 << (CtxSize+1)) - 1;
  static constexpr unsigned LayerSizes [] = { InContextSize, LS..., 1 };
public:
  BitRNAModel() : 
  mBuffer(CtxSize),
  random_generator(195486732)
  {
    for(unsigned i = 0; i < LayerCount; ++i){
      cacheResult[i].resize(1, LayerSizes[i+1]);
      cacheDerivative[i].resize(1, LayerSizes[i+1]);
    }
    for(unsigned i = 0; i < LayerCount; ++i){
      delta[i].resize(LayerSizes[i+1]);
    }

    std::uniform_real_distribution<double> distribution(-0.6, 0.6);
    for(unsigned i = 0; i < LayerCount; ++i) {
      layers[i].resize(LayerSizes[i], LayerSizes[i + 1]);
      layers[i].init([&](unsigned, unsigned) -> FixedPoint20 {
        return FixedPoint20(distribution(random_generator));
      });
    }
  }

  FixedPoint20 activation_function(FixedPoint20 const& x){
    // std::cout << x.asDouble() << " " << (-x).asDouble() << " " << (-x).exp().asDouble() << " " << (FixedPoint20::Unit() / (FixedPoint20::Unit() + (-x).exp())).asDouble() << std::endl;
    return FixedPoint20::Unit() / (FixedPoint20::Unit() + (-x).exp());
  }
  FixedPoint20 activation_derivative(FixedPoint20 const& x){
    return activation_function(x) * (FixedPoint20::Unit() - activation_function(x));
  }

  void iterateOnContext(std::function<void(unsigned)> const& f){
    f(0);
    unsigned offset = 0;
    for(unsigned i = 0; i < mBuffer.size(); ++i){
      offset = 2 * offset + mBuffer[i];
      f((1<<i) + offset);
    }
  }

  virtual std::uint32_t predict() override {
    // The input vector does not need to be computed
    /* The first vector can be computed faster than others as most in_vec components are zero */
    {
      cacheResult[0].reset();
      iterateOnContext([&](unsigned i){
        for(unsigned j = 0; j < LayerSizes[1]; ++j){
          cacheResult[0].at(j, 0) += layers[0].at(j, i);
        }
      });
      cacheDerivative[0].init([&](unsigned i, unsigned j) -> FixedPoint20 {
        return activation_derivative(cacheResult[0].at(i, j));
      });
      cacheResult[0].apply([&](unsigned, unsigned, FixedPoint20 const& v) -> FixedPoint20 {
        return activation_function(v);
      });
    }
    for(unsigned i = 1; i < LayerCount; ++i){
      auto tmp = layers[i] * cacheResult[i-1];
      cacheResult[i].init([&](unsigned i, unsigned j) -> FixedPoint20 {
        return activation_function(tmp.at(i, j));
      });
      cacheDerivative[i].init([&](unsigned i, unsigned j) -> FixedPoint20 {
        return activation_derivative(tmp.at(i, j));
      });
    }
    // --- Get prediction
    FixedPoint20 fp_prediction = cacheResult[LayerCount-1].at(0, 0);
    std::uint32_t prediction = fp_prediction.value() << 12;
    return prediction;
  }

  void train(bool b) {
    FixedPoint20 training_rate = FixedPoint20(0.3);
    // --- Create except vector ---
    Matrix<FixedPoint20> except(1, 1, [&](unsigned, unsigned) -> FixedPoint20 { return b ? FixedPoint20::Unit() : FixedPoint20(); });

    // --- First delta ---
    for(int i = 0; i < 1; ++i){
      delta[LayerCount - 1][i] = (cacheResult[LayerCount-1].at(i, 0) - except.at(i, 0)) * cacheDerivative[LayerCount-1].at(i, 0);
    }
    // --- Backpropagation deltas ---
    for(int l = LayerCount - 2; l >= 0; --l){
      for(int i = 0; i < LayerSizes[l+1]; ++i){
        delta[l][i] = 0;
        for(int n = 0; n < LayerSizes[l+2]; ++n){
          delta[l][i] += layers[l+1].at(n, i) * delta[l+1][n];
        }
        delta[l][i] *= cacheResult[l].at(i, 0);
      }
    }

    // --- Only update needed values for the first layer
    iterateOnContext([&](unsigned i){
      for(int j = 0; j < layers[0].height(); ++j){
        layers[0].at(j, i) -= training_rate * delta[0][j];
      }
    });
    for(int l = 1; l < LayerCount; ++l){
      for(int i = 0; i < layers[l].height(); ++i){
        for(int j = 0; j < layers[l].width(); ++j){
          layers[l].at(i, j) -= training_rate * delta[l][i] * cacheDerivative[l-1].at(j, 0);
        }
      }
    }
  }

  virtual void update(bool b) override {
    // --- Train network / update weighs
    train(b);
    // --- Manage context
    mBuffer.push_back(b);
  }

private:
  Matrix<FixedPoint20> layers[LayerCount];
  CircularBuffer<bool> mBuffer;
  std::default_random_engine random_generator;

  // --- Training data ---
  Matrix<FixedPoint20> cacheResult[LayerCount];
  Matrix<FixedPoint20> cacheDerivative[LayerCount];
  std::vector<FixedPoint20> delta[LayerCount];
};

template<unsigned CtxSize, unsigned ...LS>
constexpr unsigned BitRNAModel<CtxSize, LS...>::LayerCount;
template<unsigned CtxSize, unsigned ...LS>
constexpr unsigned BitRNAModel<CtxSize, LS...>::InContextSize;
template<unsigned CtxSize, unsigned ...LS>
constexpr unsigned BitRNAModel<CtxSize, LS...>::LayerSizes [];