#pragma once

#include <cinttypes>
#include <memory>
#include <iostream>

#include "FixedPoint.h"

class Model {
public:
  virtual ~Model(){ }

  virtual std::uint32_t predict(bool nxt) = 0;
};

class ConstModel : public Model{
public:
  ConstModel(std::uint32_t v = 1 << 30) : mPrediction(v){ }
  virtual ~ConstModel(){ }

  virtual std::uint32_t predict(bool nxt) override {
    return mPrediction;
  }

private:
  std::uint32_t mPrediction;
};

class MixModel : public Model{
public:
  MixModel(std::vector<Model*> models, std::vector<FixedPoint24> weigths = {}, FixedPoint24 rate = FixedPoint24(0.5)) : 
  mModels(std::move(models)),
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

  virtual std::uint32_t predict(bool nxt) override{
    std::vector<std::uint64_t> modelPredictions(mModels.size());
    std::uint64_t thisPred64 = 0;
    for(unsigned i = 0; i < mModels.size(); ++i){
      modelPredictions[i] = mModels[i]->predict(nxt); // is a 0 - 32 FixedPoint
      thisPred64 += modelPredictions[i] * mWeights[i].value(); // 8 - 56 FixedPoint
      FixedPoint24 err = (nxt ? FixedPoint24::FromValue(modelPredictions[i] >> 8) : FixedPoint24(1) - FixedPoint24::FromValue(modelPredictions[i] >> 8));
      std::cout << modelPredictions[i] << " " << mWeights[i].value() << " " << err.value() << " " << (mRate * err * err * mWeights[i]).value() << std::endl;
      mWeights[i] += mRate * err * err * mWeights[i];
    }
    std::uint32_t thisPred = static_cast<std::uint32_t>(thisPred64 >> 24); // Shift the point to ? - 32
    renormalize();
    return thisPred;
  }

private:
  std::vector<Model*> mModels;
  std::vector<FixedPoint24> mWeights;
  FixedPoint24 mRate;
};
