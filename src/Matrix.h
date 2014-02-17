#pragma once

#include <cassert>
#include <vector>
#include <functional>
#include <iostream>

template<typename T>
class Matrix{
public:
  Matrix(){ }

  Matrix(unsigned w, unsigned h) :
    mWidth(w),
    mHeight(h), 
    mData(w * h){ }

  Matrix(unsigned w, unsigned h, std::function<T(unsigned, unsigned)> const& init) :
    mWidth(w),
    mHeight(h), 
    mData(w * h){
      for(unsigned i = 0; i < mHeight; ++i){
        for(unsigned j = 0; j < mWidth; ++j){
          mData[i * mWidth + j] = init(i, j);
        }
      }
    }

  void init(std::function<T(unsigned, unsigned)> const& init){
    for(unsigned i = 0; i < mHeight; ++i){
      for(unsigned j = 0; j < mWidth; ++j){
        mData[i * mWidth + j] = init(i, j);
      }
    }
  }

  void resize(unsigned w, unsigned h){
    mWidth = w;
    mHeight = h;
    mData.resize(w * h);
  }

  unsigned width() const { return mWidth; }
  unsigned height() const { return mHeight; }

  // Index access

  T const& at(unsigned i, unsigned j) const { assert(i < mHeight && j < mWidth); return mData[i * mWidth + j]; }
  T& at(unsigned i, unsigned j) { assert(i < mHeight && j < mWidth); return mData[i * mWidth + j]; }

  // Arithmetic operators

  void operator+=(Matrix<T> const& other){
    for(unsigned i = 0; i < mHeight; ++i){
      for(unsigned j = 0; j < mWidth; ++j){
        at(i, j) += other.at(i, j);
      }
    }
  }

  Matrix<T> operator+(Matrix<T> const& other) const{
    Matrix<T> tmp = *this; tmp += other; return tmp;
  }

  void operator-=(Matrix<T> const& other){
    for(unsigned i = 0; i < mHeight; ++i){
      for(unsigned j = 0; j < mWidth; ++j){
        at(i, j) += other.at(i, j);
      }
    }
  }

  Matrix<T> operator-(Matrix<T> const& other) const{
    Matrix<T> tmp = *this; tmp -= other; return tmp;
  }

  void operator*=(Matrix const& other){
    *this = (*this) * other;
  }

  Matrix<T> operator*(Matrix<T> const& other) const{
    assert(mWidth == other.mHeight);
    Matrix<T> res(other.mWidth, mHeight);
    for(unsigned i = 0; i < mHeight; ++i){
      for(unsigned j = 0; j < other.mWidth; ++j){
        res.at(i, j) = T();
        for(unsigned k = 0; k < mWidth; ++k){
          res.at(i, j) += at(i, k) * other.at(k, j);
        }
      }
    }
    return res;
  }

  // Others

  void apply(std::function<T(unsigned, unsigned, T)> const& func){
    for(unsigned i = 0; i < mHeight; ++i){
      for(unsigned j = 0; j < mWidth; ++j){
        at(i, j) = func(i, j, at(i, j));
      }
    }
  }

private:
  unsigned mWidth, mHeight;
  std::vector<T> mData;
};
