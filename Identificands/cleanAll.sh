#!/bin/bash

find ./  -iname "*~" -exec rm {} \;
find ./  -iname "*.pyc" -exec rm {} \;
find ./  -iname __pycache__ -exec rm -r {} \;
rm -rf CMakeFiles ./dist ./Identificands.egg-info 
rm -rf Makefile CMakeCache.txt cmake_install.cmake
rm -rf ./examples/resultsData/*
rm -rf ./examples/*.tsv ./examples/*.png
rm -rf ./src/External/LARPRawReader/obj
rm -rf ./src/External/LARPRawReader/docs
rm -rf ./src/External/LARPRawReader/bin
rm -rf ./src/Identificands.egg-info
rm -rf ./examples/raw/Sample4_Papaya.raw
rm -rf CMakeDoxygenDefaults.cmake CMakeDoxyfile.in
