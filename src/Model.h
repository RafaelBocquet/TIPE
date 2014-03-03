#pragma once

#include <cinttypes>
#include <cassert>
#include <memory>
#include <iostream>
#include <iomanip>
#include <bitset>
#include <array>
#include <random>

#include "FixedPoint.h"
#include "Matrix.h"
#include "CircularBuffer.h"

// --- Model ---

class Model {
public:
  virtual ~Model(){ }

  virtual std::uint32_t predict() = 0;
  virtual void update(bool nxt) = 0;
};

// --- Model ---

class ConstModel : public Model{
public:
  ConstModel(std::uint32_t v = 1 << 30) : mPrediction(v){ }
  virtual ~ConstModel(){ }

  virtual std::uint32_t predict() override{
    return mPrediction;
  }

  virtual void update(bool) override{ }

private:
  std::uint32_t mPrediction;
};

