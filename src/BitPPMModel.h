#pragma once
#include "Model.h"


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
