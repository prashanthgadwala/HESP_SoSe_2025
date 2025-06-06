// binning.cuh
#pragma once
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include "particle.cuh"

struct Particle;

/*
 * Axis-Aligned Bounding Box (AABB) for simulation domain
 */
struct AABB {
    float3 min;  // Minimum coordinates (x_min, y_min, z_min)
    float3 max;  // Maximum coordinates (x_max, y_max, z_max)
};

/*
 * Grid structure for spatial binning
 *  - origin: lower corner of the domain
 *  - cell_size: length of each cell (usually rcut)
 *  - dims: number of cells in each direction (x, y, z)
 */
struct Grid {
    float3 origin;
    float cell_size;
    int3 dims;
};

/*
 * DeviceBinningData holds all arrays needed for cell binning on the device
 *  - cell_indices: cell index for each particle (unsorted, [num_particles])
 *  - particle_indices: indices of particles sorted by cell ([num_particles])
 *  - cell_offsets: start index in sorted array for each cell ([num_cells+1])
 *  - num_cells: total number of cells
 *  - num_particles: total number of particles
 */
struct DeviceBinningData {
    int* cell_indices;       // [num_particles]
    int* particle_indices;   // [num_particles] (sorted)
    int* cell_offsets;       // [num_cells+1]
    int num_cells;
    int num_particles;
};

/*
 * Compute grid dimensions and origin for a given domain and cutoff radius
 */
Grid compute_grid(const AABB& domain, float cutoff);

/*
 * Compute the cell index for a given position in the grid
 */
__host__ __device__ int compute_cell_index(const float3& position, const Grid& grid);

/*
 * Build the binning data structures on the device:
 *  - Assign cell indices to particles
 *  - Sort particle indices by cell
 *  - Compute cell offsets for fast neighbor search
 */
void build_binning(DeviceBinningData& bin_data, const Particle* d_particles, const Grid& grid);

/*
 * Free all device memory allocated for binning data
 */
void free_binning_data(DeviceBinningData& bin_data);

__host__ __device__ int compute_cell_index(const float3& pos, const Grid& grid);
