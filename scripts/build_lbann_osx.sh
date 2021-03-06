#!/bin/bash

################################################################
# Simple script to build LBANN in OS X.
# Can be called anywhere in the LBANN project tree.
################################################################

brew_check_install()
{
    PKG=$(brew list | grep $1)
    if [ -z "$PKG" ]; then
        brew install $1
    fi
}

# Install dependencies with Homebrew
# Note: Requires sudo access. Homebrew can be downloaded from
# http://brew.sh
# and installed / updated as shown below
#   brew tap homebrew/science
#   brew tap homebrew/boneyard
#   brew update
# Check for LBANN dependencies
brew_check_install git
brew_check_install cmake
brew_check_install clang-omp  # Require OpenMP support in clang
brew_check_install gcc49      # gfortran-4.9 is compatible with clang
brew_check_install open-mpi
brew_check_install opencv
brew_check_install doxygen
brew_check_install graphviz   # Doxygen dependency
brew_check_install metis      # Elemental dependency
brew_check_install scalapack  # Elemental dependency

################################################################
# Default options
################################################################

# Parameters
CMAKE_C_COMPILER=/usr/local/bin/clang-omp
CMAKE_CXX_COMPILER=/usr/local/bin/clang-omp++
CMAKE_Fortran_COMPILER=/usr/local/bin/gfortran-4.9
MPI_C_COMPILER=/usr/local/bin/mpicc
MPI_CXX_COMPILER=/usr/local/bin/mpicxx
MPI_Fortran_COMPILER=/usr/local/bin/mpifort
Elemental_DIR=
OpenCV_DIR=/usr/local/share/OpenCV
CUDA_TOOLKIT_ROOT_DIR=
cuDNN_DIR=
CLEAN_BUILD=0
VERBOSE=0
CMAKE_INSTALL_MESSAGE=LAZY
MAKE_NUM_PROCESSES=$(($(sysctl -n hw.ncpu) + 1))
INSTALL_LBANN=0

################################################################
# Help message
################################################################

function help_message {
  local SCRIPT=$(basename ${0})
  local N=$(tput sgr0)    # Normal text
  local C=$(tput setf 4)  # Colored text
cat << EOF
Build LBANN on an LLNL LC system.
Can be called anywhere in the LBANN project tree.
Usage: ${SCRIPT} [options]
Options:
  ${C}--help${N}                  Display this help message and exit.
  ${C}--verbose${N}               Verbose output.
  ${C}--debug${N}                 Build with debug flag.
  ${C}--clean-build${N}           Clean build directory before building.
  ${C}--make-processes${N} <val>  Number of parallel processes for make.
  ${C}--install-lbann${N}         Install LBANN headers and dynamic library into the build directory.
EOF
}

################################################################
# Parse command-line arguments
################################################################

while :; do
  case ${1} in
    -h|--help)
      # Help message
      help_message
      exit 0
      ;;
    -v|--verbose)
      # Verbose output
      VERBOSE=1
      CMAKE_INSTALL_MESSAGE=ALWAYS
      ;;
    -d|--debug)
      # Debug mode
      BUILD_TYPE=Debug
      ;;
    --clean-build|--build-clean)
      # Clean build directory
      CLEAN_BUILD=1
      ;;
    -j|--make-processes)
      if [ -n "${2}" ]; then
        MAKE_NUM_PROCESSES=${2}
      else
        echo "\"${1}\" option requires a non-empty option argument" >&2
        exit 1
      fi
      ;;
    -i|--install-lbann)
      INSTALL_LBANN=1
      ;;
    -?*)
      # Unknown option
      echo "Unknown option (${1})" >&2
      exit 1
      ;;
    *)
      # Break loop if there are no more options
      break
  esac
  shift
done

################################################################
# Initialize variables
################################################################

# Build and install directories
ROOT_DIR=$(git rev-parse --show-toplevel)
BUILD_DIR=${ROOT_DIR}/build/$(hostname)
INSTALL_DIR=${BUILD_DIR}
mkdir -p ${BUILD_DIR}
mkdir -p ${INSTALL_DIR}


################################################################
# Build LBANN
################################################################

# Work in build directory
pushd ${BUILD_DIR}

  # Clean up build directory
  if [ ${CLEAN_BUILD} -ne 0 ]; then
    CLEAN_COMMAND="rm -rf ${BUILD_DIR}/*"
    if [ ${VERBOSE} -ne 0 ]; then
      echo "${CLEAN_COMMAND}"
    fi
    ${CLEAN_COMMAND}
  fi

  # Configure build with CMake
  CONFIGURE_COMMAND=$(cat << EOF
cmake \
-D CMAKE_BUILD_TYPE=Release \
-D CMAKE_INSTALL_MESSAGE=${CMAKE_INSTALL_MESSAGE} \
-D CMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
-D CMAKE_C_COMPILER=${CMAKE_C_COMPILER} \
-D CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} \
-D CMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER} \
-D MPI_C_COMPILER=${MPI_C_COMPILER} \
-D MPI_CXX_COMPILER=${MPI_CXX_COMPILER} \
-D MPI_Fortran_COMPILER=${MPI_Fortran_COMPILER} \
-D Elemental_DIR=${Elemental_DIR} \
-D OpenCV_DIR=${OpenCV_DIR} \
-D CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR} \
-D cuDNN_DIR=${cuDNN_DIR} \
-D WITH_TBINF=OFF \
-D VERBOSE=${VERBOSE} \
-D MAKE_NUM_PROCESSES=${MAKE_NUM_PROCESSES} \
${ROOT_DIR}
EOF
)
  if [ ${VERBOSE} -ne 0 ]; then
    echo "${CONFIGURE_COMMAND}"
  fi
  ${CONFIGURE_COMMAND}
  if [ $? -ne 0 ] ; then
    echo "--------------------"
    echo "CONFIGURE FAILED"
    echo "--------------------"
    exit 1
  fi

  # Build LBANN with make
  BUILD_COMMAND="make -j${MAKE_NUM_PROCESSES} VERBOSE=${VERBOSE}"
  if [ ${VERBOSE} -ne 0 ]; then
    echo "${BUILD_COMMAND}"
  fi
  ${BUILD_COMMAND}
  if [ $? -ne 0 ] ; then
    echo "--------------------"
    echo "MAKE FAILED"
    echo "--------------------"
    exit 1
  fi

  # Install LBANN with make
  if [ ${INSTALL_LBANN} -ne 0 ]; then
      INSTALL_COMMAND="make install -j${MAKE_NUM_PROCESSES} VERBOSE=${VERBOSE}"
      if [ ${VERBOSE} -ne 0 ]; then
          echo "${INSTALL_COMMAND}"
      fi
      ${INSTALL_COMMAND}
      if [ $? -ne 0 ] ; then
          echo "--------------------"
          echo "MAKE INSTALL FAILED"
          echo "--------------------"
          exit 1
      fi
  fi

  # Generate documentation with make
  DOC_COMMAND="make doc"
  if [ ${VERBOSE} -ne 0 ]; then
    echo "${DOC_COMMAND}"
  fi
  ${DOC_COMMAND}
  if [ $? -ne 0 ] ; then
    echo "--------------------"
    echo "MAKE DOC FAILED"
    echo "--------------------"
    exit 1
  fi
  
popd
