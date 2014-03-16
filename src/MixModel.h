#pragma once
#include "Model.h"

// --- MixModel ---

class MixModel : public Model{
public:
  MixModel(std::vector<Model*> models, std::vector<FixedPoint24> weigths = {}, FixedPoint24 rate = FixedPoint24(0.006)) : 
  mModels(std::move(models)),
  mModelPredictions(mModels.size()),
  mRate(rate){
    if(mModels.size() != weigths.size()){
      mWeights.resize(mModels.size(), 0);
    } else {
      mWeights = weigths;
    }
  }

  FixedPoint24 stretch(std::uint32_t p){
    FixedPoint24 fp_p = FixedPoint24::FromValue(p >> 8);
    return fp_p.subOneLn() - (FixedPoint24::Unit() - fp_p).subOneLn();
  }

  FixedPoint24 squash(FixedPoint24 p){
    auto result = FixedPoint24::Unit() / (FixedPoint24::Unit() + (-p).exp());
    if(result < FixedPoint24()){
      return FixedPoint24();
    }else{
      return result;
    }
  }

  virtual std::uint32_t predict() override{
    std::uint64_t thisPred64 = 0;
    // --- Remember the predictions for the next update
    for(unsigned i = 0; i < mModels.size(); ++i){
      mModelPredictions[i] = stretch(mModels[i]->predict()); // 8 - 24 FixedPoint
      thisPred64 += static_cast<std::int64_t>(mModelPredictions[i].value()) * static_cast<std::int64_t>(mWeights[i].value()); // 16 - 48 FixedPoint
    }
    mLastPrediction = squash(FixedPoint24::FromValue(thisPred64 >> 24));
    return mLastPrediction.value() << 8;
  }

  virtual void update(bool nxt){
    for(Model* model : mModels){
      model->update(nxt);
    }

    for(unsigned i = 0; i < mModels.size(); ++i){
      mWeights[i] += mRate * mModelPredictions[i] * ((nxt ? FixedPoint24::Unit() : FixedPoint24()) - mLastPrediction);
    }
  }

private:
  std::vector<Model*> mModels;
  std::vector<FixedPoint24> mWeights;
  FixedPoint24 mRate;

  std::vector<FixedPoint24> mModelPredictions;
  FixedPoint24 mLastPrediction;
};