#pragma once

#include <cinttypes>
#include <cassert>
#include <memory>
#include <iostream>
#include <iomanip>
#include <bitset>
#include <array>

#include "FixedPoint.h"
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

// --- MixModel ---

class MixModel : public Model{
public:
  MixModel(std::vector<Model*> models, std::vector<FixedPoint24> weigths = {}, FixedPoint24 rate = FixedPoint24(1.0)) : 
  mModels(std::move(models)),
  mModelPredictions(mModels.size()),
  mRate(rate){
    if(mModels.size() != weigths.size()){
      mWeights.resize(mModels.size(), FixedPoint24(1.0 / mModels.size()));
    } else {
      mWeights = weigths;
      renormalize();
    }
  }

  void renormalize(){
    FixedPoint24 sum;
    for(FixedPoint24& w : mWeights){
      if(w < FixedPoint24(0)){ w = FixedPoint24(0); }
      if(w > FixedPoint24(1)){ w = FixedPoint24(1); }
      sum += w;
    }
    for(FixedPoint24& w : mWeights){
      w /= sum;
    }
  }

  virtual std::uint32_t predict() override{
    std::uint64_t thisPred64 = 0;
    for(unsigned i = 0; i < mModels.size(); ++i){
      mModelPredictions[i] = mModels[i]->predict(); // is a 0 - 32 FixedPoint
      thisPred64 += mModelPredictions[i] * mWeights[i].value(); // 8 - 56 FixedPoint
    }
    std::uint32_t thisPred = static_cast<std::uint32_t>(thisPred64 >> 24); // Shift the point to ? - 32
    return thisPred;
  }

  virtual void update(bool nxt){
    for(Model* model : mModels){
      model->update(nxt);
    }
    for(unsigned i = 0; i < mModels.size(); ++i){
      FixedPoint24 err = (nxt ? FixedPoint24::FromValue(mModelPredictions[i] >> 8) : FixedPoint24(1) - FixedPoint24::FromValue(mModelPredictions[i] >> 8));
      
      mWeights[i] += mRate * err * err * mWeights[i];
    }
    renormalize();
  }

private:
  std::vector<Model*> mModels;
  std::vector<std::uint64_t> mModelPredictions;
  std::vector<FixedPoint24> mWeights;
  FixedPoint24 mRate;
};

// --- BitPPMModel ---

template<unsigned O>
class BitPPMModel : public Model {
  using ContextType = std::array<bool, O>;
  // ctx[0] is last read bit, ctx[O-1] is first read bit
  // -- BitPPMModelTree --
  class BitPPMModelTree {
  public:
    BitPPMModelTree(unsigned depth) :
    mChildren{{nullptr, nullptr}},
    mDepth(depth),
    mCount{{0, 0}}{ }

    BitPPMModelTree(BitPPMModelTree const& other) = delete;
    BitPPMModelTree(BitPPMModelTree&& other) = delete;
    BitPPMModelTree& operator=(BitPPMModelTree const& other) = delete;
    BitPPMModelTree& operator=(BitPPMModelTree&& other) = delete;

    BitPPMModelTree& child(bool b){
      if(!mChildren[b]){
        mChildren[b] = new BitPPMModelTree(mDepth + 1);
      }
      return *mChildren[b];
    }

    std::array<std::uint64_t, 2> contextCount(ContextType const& ctx){
      if(mDepth == O){
        return mCount;
      } else {
        std::array<std::uint64_t, 2> count = child(ctx[mDepth]).contextCount(ctx);
        count[0] = 0.8 * count[0] + mCount[0];
        count[1] = 0.8 * count[1] + mCount[1];
        return count;
      }
    }

    void contextIncrement(ContextType const& ctx, bool nxt){
      if(mDepth == O){
        mCount[nxt] += 1;
      } else {
        child(ctx[mDepth]).contextIncrement(ctx, nxt);
        mCount[ctx[mDepth]] += 1;
      }
    }

    bool contextDecrement(ContextType const& ctx, bool nxt){
      if(mDepth == O){
        if(mCount[nxt] != 0){
          mCount[nxt] -= 1;
          return true;
        }else{
          return false;
        }
      } else {
        if(child(ctx[mDepth]).contextDecrement(ctx, nxt)){
          mCount[ctx[mDepth]] -= 1;
        }
        return false;
      }
    }

  private:
    std::array<BitPPMModelTree*, 2> mChildren;
    unsigned mDepth;
    std::array<std::uint64_t, 2> mCount;
  };

public:
  BitPPMModel(unsigned buffersize) :
  mBuffer(buffersize),
  mContextCount(0) {
    assert(buffersize >= O + 1);
  }

  virtual std::uint32_t predict() override {
    if(mBuffer.size() >= O){
      ContextType curContext;
      for(unsigned i = 0; i < O; ++i){
        curContext[i] = mBuffer[mBuffer.size() - i - 1];
      }
      std::array<std::uint64_t, 2> const& cCount = mContextCount.contextCount(curContext);
      if(cCount[0] == 0 && cCount[1] == 0){
        // std::cout << "No context !" << std::endl;
        return 1 << 31;
      }else{
        std::uint64_t c1 = static_cast<std::uint64_t>(cCount[1]) << 32;
        std::uint64_t c0 = cCount[0] + cCount[1];
        std::uint64_t dv = c1 / c0;
        // ---
        // std::cout << "Context : ";
        for(unsigned i = 0; i < O; ++i){
          // std::cout << curContext[O-1-i];
        }// std::cout << std::endl;
        // std::cout << cCount[0] << " " << cCount[1] << " " << static_cast<std::uint32_t>(dv) << "\n";
        // ---
        return static_cast<std::uint32_t>(dv);
      }
    }else{
      // std::cout << "No context !" << std::endl;
      return 1 << 31;
    }
  }

  virtual void update(bool b) override {
    if(mBuffer.is_full()){
      ContextType oldContext;
      for(unsigned i = 0; i < O; ++i){
        oldContext[O - i - 1] = mBuffer[i];
      }
      mContextCount.contextDecrement(oldContext, mBuffer[O]);
    }
    mBuffer.push_back(b);
    if(mBuffer.size() >= O + 1){
      ContextType newContext;
      for(unsigned i = 0; i < O; ++i){
        newContext[i] = mBuffer[mBuffer.size() - i - 2];
      }
      mContextCount.contextIncrement(newContext, b);
    }
  }
private:
  CircularBuffer<bool> mBuffer;
  BitPPMModelTree mContextCount;
};

// --- BytePPMModel ---

template<unsigned O>
class BytePPMModel : public Model {
  using ContextType = std::array<unsigned char, O>;
  // ctx[0] is last read bit, ctx[O-1] is first read bit
  // -- BytePPMModelTree --
  class BytePPMModelTree {
  public:
    BytePPMModelTree(unsigned depth) :
    mDepth(depth){
      for(unsigned i = 0; i < 256; ++i){
        mChildren[i] = nullptr;
        mCount[i] = 0;
      }
    }

    BytePPMModelTree(BytePPMModelTree const& other) = delete;
    BytePPMModelTree(BytePPMModelTree&& other) = delete;
    BytePPMModelTree& operator=(BytePPMModelTree const& other) = delete;
    BytePPMModelTree& operator=(BytePPMModelTree&& other) = delete;

    BytePPMModelTree& child(unsigned char b){
      if(!mChildren[b]){
        mChildren[b] = new BytePPMModelTree(mDepth + 1);
      }
      return *mChildren[b];
    }

    std::array<std::uint64_t, 256> contextCount(ContextType const& ctx){
      if(mDepth == O){
        return mCount;
      } else {
        std::array<std::uint64_t, 256> count = child(ctx[mDepth]).contextCount(ctx);
        for(unsigned i = 0; i < 256; ++i){
          count[i] = 3 * count[i] / 2 + mCount[i];
        }
        return count;
      }
    }

    void contextIncrement(ContextType const& ctx, unsigned char nxt){
      if(mDepth == O){
        mCount[nxt] += 1;
      } else {
        child(ctx[mDepth]).contextIncrement(ctx, nxt);
        mCount[ctx[mDepth]] += 1;
      }
    }

    bool contextDecrement(ContextType const& ctx, unsigned char nxt){
      if(mDepth == O){
        if(mCount[nxt] != 0){
          mCount[nxt] -= 1;
          return true;
        }else{
          return false;
        }
      } else {
        if(child(ctx[mDepth]).contextDecrement(ctx, nxt)){
          mCount[ctx[mDepth]] -= 1;
        }
        return false;
      }
    }

  private:
    std::array<BytePPMModelTree*, 256> mChildren;
    unsigned mDepth;
    std::array<std::uint64_t, 256> mCount;
  };

public:
  BytePPMModel(unsigned buffersize) :
  mBuffer(buffersize),
  mContextCount(0) {
    assert(buffersize >= O + 1);
    mCurBit = 1 << 7;
    mCurChar = 0;
  }

  virtual std::uint32_t predict() override {
    if(mBuffer.size() >= O){
      ContextType curContext;
      for(unsigned i = 0; i < O; ++i){
        curContext[i] = mBuffer[mBuffer.size() - i - 1];
      }
      std::array<std::uint64_t, 256> const& cCount = mContextCount.contextCount(curContext);

      std::uint64_t c1 = 0;
      std::uint64_t c0 = 0;

      for(unsigned i = mCurChar; i < mCurChar + (mCurBit << 1); ++i){
        if(i & mCurBit){
          c1 += cCount[i];
        }else{
          c0 += cCount[i];
        }
      }

      // ---
      // std::cout << "Context : ";
      for(unsigned i = 0; i < O; ++i){
        // std::cout << std::hex << (int) curContext[O-1-i] << " ";
      } // std::cout << std::endl;
      // ---

      if(c0 + c1 == 0){
        // std::cout << "No context !" << std::endl;
        return 1 << 31;
      }else{
        std::uint64_t dv = (c1 << 32) / (c1 + c0);
        // ---
        // std::cout << c0 << " " << c1 << " " << dv << "\n";
        // ---
        return (c0 == 0 ? (1ull << 32) - 1 : static_cast<std::uint32_t>(dv));
      }
    }else{
      // std::cout << "No context !" << std::endl;
      return 1 << 31;
    }
  }

  virtual void update(bool b) override {
    if(b){ 
      mCurChar += mCurBit;
    }
    if(mCurBit == 1){
      if(mBuffer.is_full()){
        ContextType oldContext;
        for(unsigned i = 0; i < O; ++i){
          oldContext[O - i - 1] = mBuffer[i];
        }
        mContextCount.contextDecrement(oldContext, mBuffer[O]);
      }
      // std::cout << "Push " << mCurChar << std::endl;
      mBuffer.push_back(mCurChar);
      if(mBuffer.size() >= O + 1){
        ContextType newContext;
        for(unsigned i = 0; i < O; ++i){
          newContext[i] = mBuffer[mBuffer.size() - i - 2];
        }
        mContextCount.contextIncrement(newContext, mCurChar);
      }

      mCurBit = 1 << 7;
      mCurChar = 0;
    }else{
      mCurBit >>= 1;
    }
  }
private:
  CircularBuffer<unsigned char> mBuffer;
  unsigned char mCurChar;
  unsigned mCurBit;
  BytePPMModelTree mContextCount;
};
