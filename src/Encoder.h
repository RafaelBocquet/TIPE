#pragma once

#include <ostream>
#include <cinttypes>
#include <iostream>
#include <iomanip>

class Encoder {
public:
  Encoder(std::ostream& stream) : 
  mStream(stream), mLow(0x0000000000000000), mHigh(0xFFFFFFFFFFFFFFFF){ }

  void encode(bool bit, std::uint32_t pred){
    // std::cout << std::hex << std::setw(16) << mLow << " " << std::setw(16) << mHigh << std::endl;
    // assert(pred != 0); // 0 probability is impossible=
    // std::cout << mLow  << " " << mHigh << " " << pred << std::endl;
    assert(mLow <= mHigh);
    /*
      compute mid
      if not bit then low = mid
      else high = mid
    */
        /* mid = ((1 - pred) * mLow + pred * mHigh) / 1 */
        /* mid = mLow + pred * (mHigh - mLow) >> 16 */
    std::uint64_t range = mHigh - mLow;
    std::uint64_t mid = mLow +
      ((pred * (range & 0x00000000FFFFFFFF)) >> 32) + 
      pred * ((range & 0xFFFFFFFF00000000) >> 32);
    // std::cout << mLow  << " " << mid << " " << mHigh << " " << pred << std::endl;
    assert(mLow <= mid && mid < mHigh);
    if(bit){
      mLow = mid + 1;
    }else{
      mHigh = mid;
    }

    while(not ((mHigh ^ mLow) & 0xFF00000000000000)){
      assert(mLow <= mHigh);
      unsigned char out = mHigh >> 56;
      mStream.put(out);
      mHigh = (mHigh << 8) | 0x00000000000000FF;
      mLow = (mLow << 8);
      assert(mLow <= mHigh);
    }
  }

private:
  std::ostream& mStream;
  std::uint64_t mLow, mHigh;
};