// binning.cu
#include "binning.cuh"
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include "particle.cuh"

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

    cx = max(0, min(cx, grid.dims.x - 1));
    cy = max(0, min(cy, grid.dims.y - 1));
    cz = max(0, min(cz, grid.dims.z - 1));

    return cx + cy * grid.dims.x + cz * grid.dims.x * grid.dims.y;
}

__global__ void kernel_assign_cell_indices(
    const Particle* particles, int* cell_indices, int N, Grid grid
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float3 pos = make_float3(particles[i].position.x, 
                                particles[i].position.y, 
                                particles[i].position.z);
        cell_indices[i] = compute_cell_index(pos, grid);
    }
}

// Kernel: Mark the start of each cell in the sorted cell_indices array
__global__ void kernel_mark_cell_starts(
    const int* sorted_cell_indices, int* cell_offsets, int num_particles, int num_cells
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    
    int cell = sorted_cell_indices[i];
    // Only mark the first occurrence of each cell
    if (i == 0 || cell != sorted_cell_indices[i - 1]) {
        cell_offsets[cell] = i;
    }
    // Always set the end offset for the last cell
    if (i == num_particles - 1) {
        cell_offsets[num_cells] = num_particles;
    }
    // protect going out-of-bounds in corner
    if (cell < num_cells)
    cell_offsets[cell] = i;

}

void build_binning(DeviceBinningData& bin_data, const Particle* d_particles, const Grid& grid) {
    int blockSize = 256;
    int gridSize = (bin_data.num_particles + blockSize - 1) / blockSize;

    // Step 1: Assign cell indices
    kernel_assign_cell_indices<<<gridSize, blockSize>>>(
        d_particles, bin_data.cell_indices, bin_data.num_particles, grid
    );
    cudaDeviceSynchronize();

    // Step 2: Initialize particle_indices as a sequence [0, 1, 2, ...]
    thrust::device_ptr<int> d_particle_indices(bin_data.particle_indices);
    thrust::sequence(thrust::device, d_particle_indices, d_particle_indices + bin_data.num_particles);

    // Step 3: Sort particle indices by cell index
    thrust::device_ptr<int> d_cell_indices(bin_data.cell_indices);
    thrust::sort_by_key(
        thrust::device,
        d_cell_indices, d_cell_indices + bin_data.num_particles,
        d_particle_indices
    );

    // Step 4: Compute cell offsets (start of each cell in sorted array)
    // Clear cell_offsets (including extra element)
    cudaMemset(bin_data.cell_offsets, 0, (bin_data.num_cells + 1) * sizeof(int));

    // Mark cell starts in the sorted cell_indices array
    kernel_mark_cell_starts<<<gridSize, blockSize>>>(
        bin_data.cell_indices, bin_data.cell_offsets, bin_data.num_particles, bin_data.num_cells
    );
    cudaDeviceSynchronize();

    // Note: cell_offsets[c] gives the start index in sorted particle_indices for cell c,
    // and cell_offsets[c+1] gives the end index (exclusive).
}

void free_binning_data(DeviceBinningData& bin_data) {
    cudaFree(bin_data.cell_indices);
    cudaFree(bin_data.particle_indices);
    cudaFree(bin_data.cell_offsets);
    bin_data = {nullptr, nullptr, nullptr, 0, 0};
}