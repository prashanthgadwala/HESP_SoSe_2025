// binning.cuh
#pragma once
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include "particle.cuh"

struct Grid {
    float3 origin;
    float cell_size;
    int3 dims;
};

Grid compute_grid(const AABB& domain, float cutoff);

__host__ __device__ int compute_cell_index(const float3& position, const Grid& grid);

void compute_cell_indices(
    thrust::device_vector<float3>& positions,
    thrust::device_vector<int>& cell_indices,
    const Grid& grid
);

void sort_particles_by_cell(
    thrust::device_vector<int>& cell_indices,
    thrust::device_vector<int>& particle_indices
);
