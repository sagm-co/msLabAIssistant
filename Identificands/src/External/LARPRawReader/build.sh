#!/bin/bash

if [[ -z "$1" ]]; then
    currPath=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
else
    currPath=$1"/src/External/LARPRawReader/"
fi

cd $currPath

dotnet nuget add source $currPath/Dependences/ --name thermoRawFileReader
dotnet nuget update source thermoRawFileReader
dotnet build LARPRawReader.csproj -c Release
doxygen Doxyfile 


