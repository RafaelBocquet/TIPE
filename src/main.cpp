#include "Rna.h"
#include "Encoder.h"
#include "Matrix.h"
#include "FixedPoint.h"
#include "Model.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cinttypes>

void archive(std::string const& filename){
  std::cout << "Archiving " << filename << std::endl;
  std::ifstream file(filename);
  if(!file.good()){
    std::cout << "Can't open file " << filename << std::endl;
    return;
  }
  
  // --- Get file size ---
  file.seekg(0, std::ios::end);
  std::uint32_t file_length = file.tellg();
  file.seekg(0, std::ios::beg);

  std::cout << "Size : " << file_length << std::endl;

  // --- Open out file ---

  std::ofstream out_file(filename + ".out");
  out_file.write((char const*) &file_length, sizeof(std::uint32_t));
  Encoder encoder(out_file);

  // --- Algo ---
  unsigned char ch;

  // Rna rna({(1 << 17) - 1 + 8, 8, 2});
  /* {
    auto activation_function = [](double x) -> double { return 1.0 / (1.0 + exp(-x)); };
    auto activation_derivative = [&](double x) -> double { return activation_function(x) * (1 - activation_function(x)); };
    MLP mlp({(1 << 17) - 1 + 8, 2}, ActivationFunction {activation_function, activation_derivative, 0.0, 1.0});
  } */
  ConstModel a0(0), a1((1 << 32) - 1), mid;
  BitPPMModel<24> ppm(10000);
  MixModel model({
    &mid, &ppm
  });
  unsigned done = 0;
  while(file.good()){
    if(done % 1000 == 0) std::cout << done << std::endl;
    file >> ch;

    for(unsigned i = 0; i < 8; ++i){
      bool bit = ch & (1 << (7-i));
      std::uint32_t pred = model.predict();
      // std::cout << "prediction : " << pred << std::endl;
      encoder.encode(bit, pred);
      model.update(bit);
    }
    done++;
  }

}

void extract(std::string const& filename){
  std::cout << "Extracting " << filename << std::endl;
  std::ifstream file(filename);
  if(!file.good()){
    std::cout << "Can't open file " << filename << std::endl;
    return;
  }

  // --- Get original file length ---

  std::uint32_t file_length;
  file.read((char*) &file_length, sizeof(std::uint32_t));

  // --- Open out file ---
  std::ofstream out_file(filename + ".orig");

  for(unsigned a = 0; a < file_length; ++a){

  }
}

// --- Argument parsing ---

void help(){
  std::cout << "Help, TODO" << std::endl;
}


enum class ProgramOption {
  Help,
  Archive,
  Extract
};

int main(int argc, char** argv){
  FixedPoint24 a(1.0);
  FixedPoint24 e = a.exp();
  std::cout << std::setprecision(25) <<  e.asDouble() << std::endl;

  std::vector<std::string> args(argc - 1);
  for(unsigned i = 0; i < argc - 1; ++i){
    args[i] = std::string(argv[i + 1]);
  }
  ProgramOption option = ProgramOption::Help;
  if(!args.empty()){
    if(args[0] == "a"){
      option = ProgramOption::Archive;
    }else if(args[0] == "x"){
      option = ProgramOption::Extract;
    }
  }
  switch(option){
  case ProgramOption::Help:
    help();
    break;
  case ProgramOption::Archive:
    for(unsigned i = 1; i < args.size(); ++i){
      archive(args[i]);
    }
    break;
  case ProgramOption::Extract:
    for(unsigned i = 1; i < args.size(); ++i){
      extract(args[i]);
    }
    break;
  }
}