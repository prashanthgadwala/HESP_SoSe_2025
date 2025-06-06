#pragma once

#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <string>

struct Vector3 {
    float x, y, z;

    __host__ __device__ Vector3() : x(0), y(0), z(0) {}
    __host__ __device__ Vector3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    __host__ __device__ Vector3 operator+(const Vector3& rhs) const {
        return Vector3(x + rhs.x, y + rhs.y, z + rhs.z);
    }

    __host__ __device__ Vector3 operator-(const Vector3& rhs) const {
        return Vector3(x - rhs.x, y - rhs.y, z - rhs.z);
    }

    __host__ __device__ Vector3 operator*(float scalar) const {
        return Vector3(x * scalar, y * scalar, z * scalar);
    }

    __host__ __device__ Vector3 operator/(float scalar) const {
        return Vector3(x / scalar, y / scalar, z / scalar);
    }

    __host__ __device__ Vector3& operator+=(const Vector3& rhs) {
        x += rhs.x; y += rhs.y; z += rhs.z;
        return *this;
    }

    __host__ __device__ Vector3& operator=(const Vector3& rhs) {
        x = rhs.x; y = rhs.y; z = rhs.z;
        return *this;
    }

    __host__ __device__ float& operator[](int i) {
        if (i == 0) return x;
        else if (i == 1) return y;
        else return z;
    }

    __host__ __device__ float norm() const {
        return sqrtf(x*x + y*y + z*z);
    }

    __host__ __device__ float squaredNorm() const {
        return x*x + y*y + z*z;
    }

};

struct Particle {
    Vector3 position;
    Vector3 velocity;
    Vector3 force;
    Vector3 acceleration; 
    float mass;
};

struct Grid{

};

void load_particles_from_file(const std::string& filename, Particle*& particles, int& num_particles);

__host__ void write_vtk(const Particle* particles, int num_particles, int step, const std::string& output_dir);

__host__ void initialize_particles(Particle* particles, int num_particles, float spacing);

__host__ void print_particles(const Particle* particles, int num_particles);

__host__ void print_diagnostics(const Particle* particles, int num_particles);

__host__ void run_simulation(Particle* particles, int num_particles, float dt, float sigma, float epsilon, float rcut, float box_size);

void cleanup_simulation();

// =====================
// Device/Kernel functions
// =====================
__device__ void update_force(Particle* p);

__device__ void apply_gravity(Particle* p);

__global__ void apply_forces(Particle* particles, int num_particles);

__global__ void velocity_verlet_step1(Particle* particles, int num_particles, float dt, float box_size[]);

__global__ void velocity_verlet_step2(Particle* particles, int num_particles, float dt);

__device__ bool in_grid_bounds(int3 coord, int3 dims);

__global__ void compute_lj_forces_binned(Particle* particles, int num_particles, float sigma, float epsilon, float rcut, float box_size[], const DeviceBinningData bin_data, const Grid grid);

__global__ void compute_lj_forces(Particle* particles, int num_particles, float sigma, float epsilon, float rcut, float box_size[]);

// =====================
// Utility (if needed)
// =====================
// Add any additional declarations here


