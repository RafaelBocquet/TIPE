#pragma once
#include "Model.h"

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