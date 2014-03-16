#!/bin/sh
clang++ -std=c++11 -O3 -Wno-c++1y-extensions -DNDEBUG -I src src/main.cpp -o tipe.out