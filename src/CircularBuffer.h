#pragma once

#include <cassert>
#include <vector>
#include <type_traits>

template<typename T>
class CircularBuffer{
  static_assert(std::is_literal_type<T>::value, "");
public:
  CircularBuffer(unsigned size) : mData(size), mFront(0), mCurrentSize(0){ }

  unsigned size() const { return mCurrentSize; }

  bool is_empty() const { return mCurrentSize == 0; }
  bool is_full() const { return mCurrentSize == mData.size(); }

  T const& front() const {
    return mData[mFront];
  }

  T const& back() const {
    return mData[(mFront + mCurrentSize - 1) % mData.size()];
  }

  void push_back(T const& val){
    if(mCurrentSize == mData.size()){
      pop_front();
    }
    mCurrentSize = mCurrentSize + 1;
    mData[(mFront + mCurrentSize - 1) % mData.size()] = val;
  }
  void pop_back(){
    assert(mCurrentSize != 0);
    mCurrentSize -= 1;
  }

  void push_front(T const& val){
    mCurrentSize = std::min<unsigned>(mCurrentSize + 1, mData.size());
    if(mFront == 0){ mFront = mData.size() - 1; }
    else{ mFront -= 1; }
    mData[mFront] = val;
  }
  void pop_front(){
    assert(mCurrentSize != 0);
    mFront += 1; if(mFront == mData.size()){ mFront = 0; }
    mCurrentSize -= 1;
  }

  T operator[](unsigned i) const{
    return mData[(mFront + i) % mData.size()];
  }

private:
  std::vector<T> mData;
  unsigned mFront;
  unsigned mCurrentSize;
};
