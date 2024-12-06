#!/bin/bash

# Function to check for a parameter in the arguments
has_param() {
    local term="$1"
    shift
    for arg; do
        if [[ $arg == "$term" ]]; then
            return 0
        fi
    done
    return 1
}

# Variables
SRC_FILES="src/test/main.c src/activations/*.c src/evaluations/*.c src/optimizers/*.c src/layers/*.c src/utilities/*.c"  # Adjust according to your project structure
INCLUDE_DIRS="include/"
BUILD_DIR="build/"
OUTPUT_FILE="${BUILD_DIR}network"  # Output executable name
CFLAGS="-O3 -march=native -funroll-loops -ftree-vectorize -g -fopenmp -lm
 -I${INCLUDE_DIRS} -I${INCLUDE_DIRS}activations -I${INCLUDE_DIRS}evaluations -I${INCLUDE_DIRS}optimizers -I${INCLUDE_DIRS}layers -I${INCLUDE_DIRS}utilities"
PARALLEL_FLAG=""
DIAGNOSTIC_FLAG=""
SOCKET_FLAG=""

# Check for flags
if has_param "-parallel" "$@"; then
    echo "Compiling with OpenMP parallelization enabled..."
    PARALLEL_FLAG="-D ENABLE_PARALLEL -D NUM_THREADS=8"  # You can change NUM_THREADS based on your system
fi

if has_param "-diag" "$@"; then
    echo "Compiling with debugging diagnostics flag enabled..."
    DIAGNOSTIC_FLAG="-fsanitize=address,undefined"
fi

# Create build directory if it doesn't exist
if [[ ! -d "$BUILD_DIR" ]]; then
    echo "Creating build directory: $BUILD_DIR"
    mkdir -p "$BUILD_DIR"
fi

# Default Compilation (Executable)
echo "Compiling the program..."
clang $CFLAGS $PARALLEL_FLAG $DIAGNOSTIC_FLAG $SRC_FILES -o $OUTPUT_FILE

# Check if compilation was successful
if [[ $? -ne 0 ]]; then
    echo "Compilation failed. Exiting."
    exit 1
fi

echo "Compilation successful. Output file: $OUTPUT_FILE"

# Run the program
echo "Running the program..."
time $OUTPUT_FILE 
