# Assignment 2 - Molecular Dynamics Simulation

## Course: High-End Simulation in Practice

### Summary

This project implements a 3D molecular dynamics simulator using CUDA to learn how GPU kernels can be applied to perform complex physics simulations. The simulator calculates inter-particle forces using the **Lennard-Jones potential** and evolves the system over time using **Velocity Verlet integration**.

We simulate various configurations of meshed particles, supporting input from files and output to VTK format for visualization in ParaView.

---

## Features

- GPU-accelerated force computation using Lennard-Jones potential
- Velocity Verlet integration scheme
- Supports file-based initialization
- Generates VTK files for visualization
- Includes test cases for validation
- Benchmarking support and performance logging

---

## Input Parameters

These parameters can be configured via command-line or config file:

| Parameter         | Description |
|------------------|-------------|
| `input_file`      | Path to input file describing particle positions, velocities, and masses |
| `output_dir`      | Directory where VTK files and plots will be stored |
| `dt`              | Time step value |
| `num_steps`       | Number of simulation steps |
| `sigma`           | Lennard-Jones constant (characteristic distance) |
| `epsilon`         | Lennard-Jones constant (potential well depth) |
| `output_freq`     | Frequency at which VTK output is written |
| `benchmark`       | Enable/disable performance logging |
| `test_case`       | Select from predefined test scenarios (e.g. STABLE, ATTRACTIVE, REPULSIVE) |

---

## Output Structure

- `vtk_final_position/` – contains `.vtk` files output every `output_freq` steps, visualizable in ParaView
- `plot_final_position/` – contains plots and images of final configurations for test cases and benchmarks
- `benchmark_logs/` – optional CSV logs when benchmarking is enabled

---

## Build and Run

Make sure the `run_all.sh` script is executable and then run it:

```bash
chmod +x run_all.sh
./run_all.sh
```

This script compiles the CUDA code, runs simulations for test cases, generates VTK outputs, and saves performance logs and plots.

## Test Cases
Three core 2-particle test scenarios are included to validate correctness:

STABLE: Particles placed at equilibrium distance (zero net force).

REPULSIVE: Particles start too close and repel each other.

ATTRACTIVE: Particles start too far and are drawn together.

## Visualization
Use ParaView or any compatible VTK viewer to load the files in vtk_final_position/ and observe the particle motion and interaction over time.

## Notes
All source code is written in modern C++ with CUDA for GPU acceleration.

Modular structure: particle.cuh (definitions), particle.cu (kernel logic), main.cu (simulation driver).

Benchmarking mode logs average step times for performance analysis.

## Authors
RISHYAVANDHAN VENKATESAN - fi11maka
PRASHANTH GADWALA - yl34esew
AHEMAD DANIYAL - oq18afiw