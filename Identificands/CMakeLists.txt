cmake_minimum_required(VERSION 3.22)

project(Identificands)
set(CUSTOM_RAWREADER_DIR "${CMAKE_SOURCE_DIR}/src/External/LARPRawReader")

find_package(Python3 REQUIRED)
find_program(RAWREADER_EXECUTABLE
  NAMES LARPRawReader
  PATHS ${CUSTOM_RAWREADER_DIR}
  NO_DEFAULT_PATH
  )

find_package(Doxygen)

if(Doxygen_FOUND)
  message(STATUS "Found Doxygen executable")
else()
    message(STATUS "Doxygen not found. Documentation target will not be created.")
endif()

if(RAWREADER_EXECUTABLE)
    message(STATUS "Found LARPRawReader executable at: ${RAWREADER_EXECUTABLE}")
else()

  find_program(DOTNET_EXECUTABLE NAMES dotnet)
  if(DOTNET_EXECUTABLE)
    set(SCRIPT_PATH "${CMAKE_SOURCE_DIR}/src/External/LARPRawReader/build.sh")  
    message(STATUS "Found .NET CLI at: ${DOTNET_EXECUTABLE}")
    message(STATUS "Building LARPRawReader")

    add_custom_target(
      installAndBuildDependences
      ALL
      COMMAND bash "${SCRIPT_PATH}" "${CMAKE_CURRENT_BINARY_DIR}"
      COMMAND cd examples/raw && tar xzf Sample4_Papaya.tar.gz
      COMMENT "Building LARPRawReader"
      COMMAND ${Python3_EXECUTABLE} setup.py egg_info
      COMMAND pip install -r Identificands.egg-info/requires.txt
      #COMMAND ${Python3_EXECUTABLE} -m build
      )    
  else()
    message(FATAL_ERROR "DOTNET 6.0 or higher is required.")
  endif()  
endif()


