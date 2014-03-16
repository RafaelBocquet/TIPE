#pragma once
#include "Model.h"

// --- Rna model ---

class RNAContext {
public:
  static constexpr unsigned ContextSize = 3211519;

  RNAContext() : mCharPos(0), mCurrentChar(0), mBuffer(4) { }

  void iterateOnContext(std::function<void(unsigned)> const& f){
    static const unsigned pow2 [] = {
      1, 2, 4, 8, 16, 32, 64, 128
    };
    // 0 -> 0 Always on
    f(0);
    // 1 -> 255 Current char
    unsigned currentCharContext = pow2[mCharPos] + mCurrentChar;
    assert(1 <= currentCharContext && currentCharContext <= 255);
    f(currentCharContext);
    // 256 -> 65790 Last char
    unsigned lastCharContext = 255 + 256 * mBuffer[mBuffer.size() - 1] + pow2[mCharPos] + mCurrentChar;
    assert(256 <= lastCharContext && lastCharContext <= 65790);
    f(lastCharContext);
    // 0 -> 16777214
    unsigned long last2CharContext = 256 * 256 * mBuffer[mBuffer.size() - 2] +
      256 * mBuffer[mBuffer.size() - 1] +
      pow2[mCharPos] + mCurrentChar - 1;
    // 65791 -> 1114366
    unsigned last2CharContextHash = 65791 + last2CharContext % 1048576;
    f(last2CharContextHash);
    // 0 -> 4294967296
    unsigned long last3CharContext =
      256 * 256 * 256 * mBuffer[mBuffer.size() - 3] +
      256 * 256 * mBuffer[mBuffer.size() - 2] +
      256 * mBuffer[mBuffer.size() - 1] +
      pow2[mCharPos] + mCurrentChar - 1;
    // 1114367 -> 2162942
    unsigned last3CharContextHash = 1114367 + last3CharContext % 1048576;
    f(last3CharContextHash);
    // 0 -> 1099511627776
    unsigned long last4CharContext =
      256 * 256 * 256 * 256 * mBuffer[mBuffer.size() - 4] +
      256 * 256 * 256 * mBuffer[mBuffer.size() - 3] +
      256 * 256 * mBuffer[mBuffer.size() - 2] +
      256 * mBuffer[mBuffer.size() - 1] +
      pow2[mCharPos] + mCurrentChar - 1;
    // 2162943 -> 3211518
    unsigned last4CharContextHash = 1114367 + last3CharContext % 1048576;
    f(last3CharContextHash);
  }

  void update(bool bit){
    if(bit){
      mCurrentChar |= (1 << mCharPos);
    }
    mCharPos += 1;
    if(mCharPos == 8){
      mBuffer.push_back(mCurrentChar);
      mCurrentChar = 0;
      mCharPos = 0;
    }
  }

private:
  unsigned mCharPos;
  unsigned char mCurrentChar;
  CircularBuffer<unsigned char> mBuffer;
};

template<typename Ctx>
class RNAModel : public Model {
public:
  RNAModel() :
  random_generator(195486732),
  mContext()
  {
    std::uniform_real_distribution<double> distribution(-0.6, 0.6);
    mMatrix.resize(Ctx::ContextSize, 1);
    mMatrix.init([&](unsigned, unsigned) -> FixedPoint20 {
      return FixedPoint20(distribution(random_generator));
    });
  }

  FixedPoint20 activation_function(FixedPoint20 const& x){
    return FixedPoint20::Unit() / (FixedPoint20::Unit() + (-x).exp());
  }
  FixedPoint20 activation_derivative(FixedPoint20 const& x){
    return activation_function(x) * (FixedPoint20::Unit() - activation_function(x));
  }

  virtual std::uint32_t predict() override {
    mResult = FixedPoint20();
    mContext.iterateOnContext([&](unsigned i){
      mResult += mMatrix.at(0, i);
    });
    mDerivative = activation_derivative(mResult);
    mResult = activation_function(mResult);

    std::uint32_t prediction = mResult.value() << 12;
    return prediction;
  }

  void train(bool b) {
    FixedPoint20 training_rate = FixedPoint20(0.3);
    // --- Create except vector ---
    FixedPoint20 delta = (mResult - (b ? FixedPoint20(1.0) : FixedPoint20(0.0))) * mDerivative;

    mContext.iterateOnContext([&](unsigned i){
      mMatrix.at(0, i) -= training_rate * delta;
    });
  }

  virtual void update(bool b) override {
    // --- Train network / update weighs
    train(b);
    // --- Manage context
    mContext.update(b);
  }

private:
  Ctx mContext;
  Matrix<FixedPoint20> mMatrix;
  std::default_random_engine random_generator;
  FixedPoint20 mResult, mDerivative;
};