#pragma once

#include <cinttypes>
#include <iostream>

template<unsigned IB, unsigned FB>
class FixedPoint {
  static_assert(IB + FB == 32, "");
  static constexpr std::int32_t unit = 1 << FB;
  static constexpr std::int32_t zero = 0;
  static constexpr std::int32_t maximum = 0x7FFFFFFF;

  using SelfType = FixedPoint<IB, FB>;

public:
  constexpr FixedPoint() : mValue(zero){ }
  constexpr FixedPoint(double const& d) : mValue(static_cast<std::int32_t>(d * unit)){ }
  constexpr FixedPoint(float const& d) : mValue(static_cast<std::int32_t>(d * unit)){ }
  constexpr FixedPoint(int const& d) : mValue(static_cast<std::int32_t>(d * unit)){ }

  double asDouble() const{
    return ((double) mValue) / ((double) unit);
  }
  
  static constexpr SelfType FromValue(std::int32_t value){
    SelfType r = SelfType(); r.mValue = value; return r;
  }

  static constexpr SelfType Unit(){
    return FromValue(unit);
  }


  std::int32_t value() const {
    return mValue;
  }

  void operator+=(SelfType const& other){
    mValue += other.mValue;
  }
  SelfType operator+(SelfType const& other) const{
    SelfType tmp = *this; tmp += other; return tmp;
  }

  void operator-=(SelfType const& other){
    mValue -= other.mValue;
  }
  SelfType operator-(SelfType const& other) const{
    SelfType tmp = *this; tmp -= other; return tmp;
  }

  SelfType operator-() const{
    SelfType tmp = *this; tmp.mValue *= -1; return tmp;
  }

  SelfType operator*(SelfType const& other) const{
    SelfType tmp;
    std::int64_t a = mValue, b = other.mValue;
    tmp.mValue = static_cast<std::int32_t>((a * b) >> FB);
    return tmp;
  }
  void operator*=(SelfType const& other){
    *this = *this * other;
  }

  SelfType operator/(SelfType const& other) const{
    if(other.mValue == 0){
      if(mValue > 0){
        return SelfType::FromValue(maximum);
      }else if(mValue == 0){
        return SelfType::FromValue(0);
      }else{
        return SelfType::FromValue(-(maximum - 1));
      }
    }
    SelfType tmp;
    std::int64_t a = mValue, b = other.mValue;
    tmp.mValue = static_cast<std::int32_t>((a << FB) / b);
    return tmp;
  }
  void operator/=(SelfType const& other){
    *this = *this / other;
  }

  bool operator==(SelfType const& other) const { return mValue == other.mValue; }
  bool operator!=(SelfType const& other) const { return mValue != other.mValue; }
  bool operator<(SelfType const& other) const { return mValue <= other.mValue; }
  bool operator<=(SelfType const& other) const { return mValue <= other.mValue; }
  bool operator>(SelfType const& other) const { return mValue > other.mValue; }
  bool operator>=(SelfType const& other) const { return mValue >= other.mValue; }

  SelfType exp() const;

private:
  std::int32_t mValue;
};

template<unsigned IB, unsigned FB>
constexpr std::int32_t FixedPoint<IB, FB>::unit;
template<unsigned IB, unsigned FB>
constexpr std::int32_t FixedPoint<IB, FB>::zero;
template<unsigned IB, unsigned FB>
constexpr std::int32_t FixedPoint<IB, FB>::maximum;

#include "FixedPointExp.inl"

using FixedPoint24 = FixedPoint<8, 24>;
using FixedPoint20 = FixedPoint<12, 20>;