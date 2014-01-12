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

  static FixedPoint FromValue(std::int32_t value){
    FixedPoint r; r.mValue = value; return r;
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

#include "FixedPointExp.inl"

using FixedPoint24 = FixedPoint<8, 24>;