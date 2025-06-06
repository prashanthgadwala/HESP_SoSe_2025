#pragma once
#include "binning.cuh"

struct DeviceNeighborData {
    int* neighbors;       // Flattened neighbor lists
    int* num_neighbors;   // Number of neighbors per particle
    float* rcut_sq;       // Squared cutoff distance
    int max_neighbors;    // Max neighbors per particle
    int num_particles;    // Number of particles
};

void build_neighbor_list(DeviceNeighborData& nb_data, const Particle* d_particles, const DeviceBinningData& bin_data, const Grid& grid, float rcut, const float box_size[3]);

void free_neighbor_data(DeviceNeighborData& nb_data);