#pragma once

#include <ostream>
#include <cinttypes>
#include <iostream>
#include <iomanip>

class Encoder {
public:
  Encoder(std::ostream& stream) : 
  mStream(stream), mLow(0x0000000000000000), mHigh(0xFFFFFFFFFFFFFFFF){ }

  ~Encoder(){
    unsigned char out1 = mLow >> 56;
    unsigned char out2 = mLow >> 48;
    unsigned char out3 = mLow >> 40;
    unsigned char out4 = mLow >> 32;
    unsigned char out5 = mLow >> 24;
    unsigned char out6 = mLow >> 16;
    unsigned char out7 = mLow >> 8;
    unsigned char out8 = mLow;
    mStream.put(out1);
    mStream.put(out2);
    mStream.put(out3);
    mStream.put(out4);
    mStream.put(out5);
    mStream.put(out6);
    mStream.put(out7);
    mStream.put(out8);
  }

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
    std::cout << std::hex << mLow  << " " << mid << " " << mHigh << " " << pred << std::endl;
    assert(mLow <= mid && mid < mHigh);
    if(not bit){
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