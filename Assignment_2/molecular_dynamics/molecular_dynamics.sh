#!/bin/bash

mkdir -p build
cd build || exit 1
rm -rf *
cmake ..
make -j
cd ..

# If --benchmark-all is passed, run the benchmark script
if [[ "$1" == "--benchmark-all" ]]; then
    echo "Running full benchmark sweep..."
    
    # CONFIGURABLE
    EXEC=./build/molecular_dynamics
    OUTPUT_DIR=output
    INPUT_DIR=input
    DT=0.005
    STEPS=100
    SIGMA=1.0
    EPSILON=1.0

    mkdir -p $OUTPUT_DIR

    for input_file in $INPUT_DIR/particles_*.txt; do
        echo "Benchmarking $input_file..."
        $EXEC \
            --input_file "$input_file" \
            --output_dir "$OUTPUT_DIR" \
            --dt $DT \
            --steps $STEPS \
            --sigma $SIGMA \
            --epsilon $EPSILON \
            --benchmark
    done

else
    # Default run (single test case, no benchmarking)
    ./build/molecular_dynamics
fi
