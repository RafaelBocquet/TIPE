#pragma once

#include <ostream>
#include <cinttypes>
#include <iostream>
#include <iomanip>
#include <cassert>

class Decoder {
public:
  Decoder(std::istream& stream) : 
  mStream(stream), mLow(0x0000000000000000), mHigh(0xFFFFFFFFFFFFFFFF), mActual(0x0000000000000000){
    for(unsigned i = 0; i < 8 && stream.good(); ++i){
      mStream.get(*(reinterpret_cast<char *>(&mActual) + (7-i)));
    }
  }

  ~Decoder(){
  }

  bool decode(std::uint32_t pred){
    assert(mLow <= mHigh);
    /*
      compute mid
      if not bit then low = mid
      else high = mid
    */

    /* mid = ((1 - pred) * mLow + pred * mHigh) / 1 */
    /* mid = mLow + pred * (mHigh - mLow) >> 16 */

    /* 0 case : 
      mid = mLow
      */
    /* 1 case :
      mid = mHigh
      */

    std::uint64_t range = mHigh - mLow;
    std::uint64_t mid = mLow +
      ((pred * (range & 0x00000000FFFFFFFF)) >> 32) + 
      pred * ((range & 0xFFFFFFFF00000000) >> 32);

    assert(mLow <= mid && mid < mHigh);

    bool bit = mActual <= mid;
    if(not bit){
      mLow = mid + 1;
    }else{
      mHigh = mid;
    }

    while(not ((mHigh ^ mLow) & 0xFF00000000000000)){
      assert(mLow <= mHigh);
      mHigh = (mHigh << 8) | 0x00000000000000FF;
      mLow = (mLow << 8);
      mActual = (mActual << 8);
      mStream.get(*reinterpret_cast<char *>(&mActual));

      assert(mLow <= mHigh);
    }

    return bit;
  }

private:
  std::istream& mStream;
  std::uint64_t mLow, mHigh;
  std::uint64_t mActual;
};