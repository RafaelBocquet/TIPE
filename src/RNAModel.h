#pragma once
#include "Model.h"

// --- Rna model ---

class RNAContext {
public:
  static constexpr unsigned ContextSize = 67174647;

  RNAContext() : mCharPos(0), mCurrentChar(0), mBuffer(4) { }

  void iterateOnContext(std::function<void(unsigned)> const& f){
    static const unsigned pow2 [] = {
      1, 2, 4, 8, 16, 32, 64, 128
    };
    // 0 -> 0 Always on
    // Note : does not improve compression
    // f(0);
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
    // 65791 -> 16843005
    unsigned last2CharContextHash = 65791 + last2CharContext % 16777214;
    f(last2CharContextHash);
    // 0 -> 4294967296
    unsigned long last3CharContext =
      256 * 256 * 256 * mBuffer[mBuffer.size() - 3] +
      256 * 256 * mBuffer[mBuffer.size() - 2] +
      256 * mBuffer[mBuffer.size() - 1] +
      pow2[mCharPos] + mCurrentChar - 1;
    // 16843006 -> 33620219
    unsigned last3CharContextHash = 16843006 + last3CharContext % 16777214;
    f(last3CharContextHash);
    // 0 -> 1099511627776
    unsigned long last4CharContext =
      256 * 256 * 256 * 256 * mBuffer[mBuffer.size() - 4] +
      256 * 256 * 256 * mBuffer[mBuffer.size() - 3] +
      256 * 256 * mBuffer[mBuffer.size() - 2] +
      256 * mBuffer[mBuffer.size() - 1] +
      pow2[mCharPos] + mCurrentChar - 1;
    // 33620219 -> 50397433
    unsigned last4CharContextHash = 33620219 + last4CharContext % 16777214;
    f(last4CharContextHash);
    unsigned long last5CharContext =
      256 * 256 * 256 * 256 * 256 * mBuffer[mBuffer.size() - 5] +
      256 * 256 * 256 * 256 * mBuffer[mBuffer.size() - 4] +
      256 * 256 * 256 * mBuffer[mBuffer.size() - 3] +
      256 * 256 * mBuffer[mBuffer.size() - 2] +
      256 * mBuffer[mBuffer.size() - 1] +
      pow2[mCharPos] + mCurrentChar - 1;
    // 50397433 -> 67174647
    unsigned last5CharContextHash = 50397433 + last5CharContext % 16777214;
    f(last5CharContextHash);
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
  mContext()
  {
    mMatrix.resize(Ctx::ContextSize, 1);
    mMatrix.init([&](unsigned, unsigned) -> FixedPoint20 {
      return FixedPoint20(0);
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
    FixedPoint20 training_rate = FixedPoint20(0.375);
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