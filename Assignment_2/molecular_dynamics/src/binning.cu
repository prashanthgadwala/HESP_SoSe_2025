// binning.cu
#include "binning.cuh"
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

// Compute number of cells in each dimension based on cutoff
Grid compute_grid(const AABB& domain, float cutoff) {
    Grid grid;
    grid.origin = domain.min;
    grid.cell_size = cutoff;
    grid.dims = make_int3(
        static_cast<int>(ceil((domain.max.x - domain.min.x) / cutoff)),
        static_cast<int>(ceil((domain.max.y - domain.min.y) / cutoff)),
        static_cast<int>(ceil((domain.max.z - domain.min.z) / cutoff))
    );
    return grid;
}

__host__ __device__ int compute_cell_index(const float3& pos, const Grid& grid) {
    int cx = static_cast<int>((pos.x - grid.origin.x) / grid.cell_size);
    int cy = static_cast<int>((pos.y - grid.origin.y) / grid.cell_size);
    int cz = static_cast<int>((pos.z - grid.origin.z) / grid.cell_size);

    // Clamp to grid bounds
    cx = max(0, min(cx, grid.dims.x - 1));
    cy = max(0, min(cy, grid.dims.y - 1));
    cz = max(0, min(cz, grid.dims.z - 1));

    return cx + cy * grid.dims.x + cz * grid.dims.x * grid.dims.y;
}

// Kernel to compute cell indices
__global__ void kernel_compute_cell_indices(
    float3* positions, int* cell_indices, int N, Grid grid
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        cell_indices[i] = compute_cell_index(positions[i], grid);
    }
}

void compute_cell_indices(
    thrust::device_vector<float3>& positions,
    thrust::device_vector<int>& cell_indices,
    const Grid& grid
) {
    int N = positions.size();
    int blockSize = 128;
    int gridSize = (N + blockSize - 1) / blockSize;

    kernel_compute_cell_indices<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(positions.data()),
        thrust::raw_pointer_cast(cell_indices.data()),
        N, grid
    );
    cudaDeviceSynchronize();
}

// Sort particles by their cell index
void sort_particles_by_cell(
    thrust::device_vector<int>& cell_indices,
    thrust::device_vector<int>& particle_indices
) {
    thrust::sort_by_key(
        cell_indices.begin(), cell_indices.end(), particle_indices.begin()
    );
}
