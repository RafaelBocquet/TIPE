#include "Encoder.h"
#include "Decoder.h"
#include "Matrix.h"
#include "FixedPoint.h"
#include "Model.h"
#include "BitRNAModel.h"
#include "RNAModel.h"
#include "BytePPMModel.h"
#include "BitPPMModel.h"
#include "MixModel.h"

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
  char ch;

  RNAModel<RNAContext> model;
  unsigned done = 0;
  while(file.good()){
    if(done % 10000 == 0) std::cout << done << std::endl;
    file.get(ch);

    for(unsigned i = 0; i < 8; ++i){
      bool bit = ch & (1 << (7-i));
      std::uint32_t pred = model.predict();
      // std::cout << "prediction : " << pred << " " << std::setprecision(10) << (double) pred / (double) (1ull << 32) << " " << bit << std::endl;
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
  Decoder decoder(file);

  // --- Open out file ---
  std::ofstream out_file(filename + ".orig");

  // --- Algo ---
  BitRNAModel<16> rna;

  for(unsigned a = 0; a < file_length; ++a){
    char ch = 0;
    for(unsigned i = 0; i < 8; ++i){
      std::uint32_t pred = rna.predict();
      bool bit = decoder.decode(pred);
      if(bit){
        ch |= 1 << (7-i);
      }
      rna.update(bit);
    }
    out_file.put(ch);
  }
}

void html_bpc(std::string const& filename){
  std::cout << "html bpc for : " << filename << std::endl;
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

  std::ofstream out_file(filename + ".html");

  // --- Algo ---
  char ch;

  RNAModel<RNAContext> model;
  unsigned done = 0;
  while(file.good() && !file.eof()){
    if(done % 10000 == 0) std::cout << done << std::endl;
    file.get(ch);

    double bitcount = 0.0;

    for(unsigned i = 0; i < 8; ++i){
      bool bit = ch & (1 << (7-i));
      std::uint32_t pred = model.predict();
      double pred_double = static_cast<double>(pred) / static_cast<double>((std::uint64_t) 1 << 32);
      double err = bit ? 1.0 - pred_double : pred_double;
      bitcount -= std::log(1.0 - err) / std::log(2.0);

      model.update(bit);
    }
    int r = 255.0 * bitcount / 8.0;
    int g = 255.0 * (1.0 - bitcount / 8.0);
    int b = 0;
    out_file << "<span style=\"background-color:rgb(" << r << ", " << g << ", " << b << ");\">" << ch << "</span>";
    if(ch == '\n'){
      out_file << "<br/>";
    }
    done++;
  }
}

// --- Argument parsing ---

void help(){
  std::cout << "Help, TODO" << std::endl;
}


enum class ProgramOption {
  Help,
  Archive,
  Extract,
  HTMLBPC
};

int main(int argc, char** argv){
  std::vector<std::string> args(argc - 1);
  for(unsigned i = 0; i < argc - 1; ++i){
    args[i] = std::string(argv[i + 1]);
  }
  ProgramOption option = ProgramOption::Help;
  if(!args.empty()){
    if(args[0] == "a"){
      option = ProgramOption::Archive;
    }if(args[0] == "b"){
      option = ProgramOption::HTMLBPC;
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
  case ProgramOption::HTMLBPC:
    for(unsigned i = 1; i < args.size(); ++i){
      html_bpc(args[i]);
    }
    break;
  case ProgramOption::Extract:
    for(unsigned i = 1; i < args.size(); ++i){
      extract(args[i]);
    }
    break;
  }
}