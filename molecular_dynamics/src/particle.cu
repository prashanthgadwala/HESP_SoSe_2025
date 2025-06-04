#include <iostream>
#include "particle.cuh"
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>


void load_particles_from_file(const std::string& filename, Particle*& particles, int& num_particles) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Cannot open input file " + filename);
    }

    std::string line;
    std::getline(file, line);  // Skip header line

    std::vector<Particle> particle_p;
    float x, y, z, vx, vy, vz, m;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;  // Skip empty/comment lines
        std::istringstream iss(line);
        if (!(iss >> x >> y >> z >> vx >> vy >> vz >> m)) continue;

        Particle p;
        p.position = Vector3(x, y, z);
        p.velocity = Vector3(vx, vy, vz);
        p.force = Vector3(0.0f, 0.0f, 0.0f);
        p.mass = m;
        particle_p.push_back(p);
    }

    file.close();

    num_particles = particle_p.size();
    particles = new Particle[num_particles];
    for (int i = 0; i < num_particles; ++i) {
        particles[i] = particle_p[i];
    }
}


__host__ void initialize_particles(Particle* particles, int num_particles, float spacing) {
    for (int i = 0; i < num_particles; ++i) {
        particles[i].position = Vector3(0.05f + i * spacing, 0.05f + i * spacing, 0.05f + i * spacing);
        particles[i].velocity = Vector3(0.0f, 0.0f, 0.0f);
        particles[i].force = Vector3(0.0f, 0.0f, 0.0f);
        particles[i].mass = 10.0f;
    }
}

__device__ void update_force(Particle* p) {
    p->force = Vector3(0.0f, 0.0f, 0.0f);
}

__device__ void apply_gravity(Particle* p) {
    Vector3 gravity(0.0f, -9.81f, 0.0f);
    p->force += gravity * p->mass;
}

__global__ void apply_forces(Particle* particles, int num_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles) {
        update_force(&particles[idx]);
        apply_gravity(&particles[idx]);
    }
}

__host__ void print_particles(const Particle* particles, int num_particles) {
    for (int i = 0; i < num_particles; ++i) {
        const auto& p = particles[i];
        std::cout << "Particle " << i
                  << " | Pos: (" << p.position.x << ", " << p.position.y << ", " << p.position.z << ")"
                  << " | Vel: (" << p.velocity.x << ", " << p.velocity.y << ", " << p.velocity.z << ")"
                  << " | Force: (" << p.force.x << ", " << p.force.y << ", " << p.force.z << ")"
                  << " | Mass: " << p.mass << '\n';
    }
    std::cout<<std::endl;
}


__global__ void velocity_verlet_step1(Particle* particles, int num_particles, float dt, float box_size[]) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_particles) {
        auto& p = particles[i];
        // Store current acceleration
        p.acceleration = (p.force / p.mass);
        
        // Update position: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
        p.position += p.velocity * dt + p.acceleration * (0.5f * dt * dt);   

        // Apply periodic boundary conditions
        for (int d = 0; d < 3; ++d) {
            if (p.position[d] < 0.0f)
                p.position[d] += box_size[d];
            else if (p.position[d] >= box_size[d])
                p.position[d] -= box_size[d];
        }
    }
}

__global__ void velocity_verlet_step2(Particle* particles, int num_particles, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_particles) {
        auto& p = particles[i];

        Vector3 a_new = p.force / p.mass;
        
        // Update velocity: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        p.velocity += (p.acceleration + a_new) * (0.5f * dt);
        
        // Store new acceleration for next step
        p.acceleration = a_new;
    }
}

__global__ void compute_lj_forces(Particle* particles, int num_particles, float sigma, float epsilon, float rcut, float box_size[]) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    Vector3 total_force(0.0f, 0.0f, 0.0f);

    float rcut_sq = rcut * rcut;

    for (int j = 0; j < num_particles; ++j) {
        if ((i != j)) 
        {
            Vector3 rij = particles[j].position - particles[i].position;
            // Apply minimum image convention
            for (int d = 0; d < 3; ++d) {
                float box_d = box_size[d];
                if (rij[d] >  0.5f * box_d) rij[d] -= box_d;
                if (rij[d] < -0.5f * box_d) rij[d] += box_d;
            }

            float r2 = rij.squaredNorm();
            if (rcut == 0.0f || r2 < rcut_sq)
            {
                float r = sqrtf(r2);
                float inv_r2 = 1.0f / r2;
                float sigma2 = sigma * sigma;
                float term = sigma2 * inv_r2;          // (sigma/r)^2
                float B_m = term * term * term;        // (sigma/r)^6
                float A_n = B_m * B_m;                 // (sigma/r)^12
                float f_mag = (24.0f * epsilon * (2.0f * A_n - B_m)) * inv_r2;
                Vector3 f_dir = rij / r;
                total_force += f_dir * f_mag;
            }
        }
    }
    particles[i].force = total_force;
}

__host__ void run_simulation(Particle* particles, int num_particles, float dt, float sigma, float epsilon, float rcut, float box_size[]) {
    Particle* d_particles;
    size_t size = num_particles * sizeof(Particle);
    cudaMalloc(&d_particles, size);
    cudaMemcpy(d_particles, particles, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (num_particles + blockSize - 1) / blockSize;

    if (dt > 0.0f) {
        // Step 1: Position update only if dt is non-zero
        velocity_verlet_step1<<<gridSize, blockSize>>>(d_particles, num_particles, dt, box_size);
        cudaDeviceSynchronize();
    }

    // Recompute forces (always done)
    //apply_forces<<<gridSize, blockSize>>>(d_particles, num_particles);
    compute_lj_forces<<<gridSize, blockSize>>>(d_particles, num_particles, sigma, epsilon, rcut, box_size);
    cudaDeviceSynchronize();

    if (dt > 0.0f) {
        // Step 3: Velocity update
        velocity_verlet_step2<<<gridSize, blockSize>>>(d_particles, num_particles, dt);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(particles, d_particles, size, cudaMemcpyDeviceToHost);
    cudaFree(d_particles);
}



__host__ void print_diagnostics(const Particle* particles, int num_particles) {
    Vector3 total_momentum(0.0f, 0.0f, 0.0f);
    float total_kinetic_energy = 0.0f;
    float max_velocity = 0.0f;
    float max_position = 0.0f;

    for (int i = 0; i < num_particles; ++i) {
        const Particle& p = particles[i];

        total_momentum += p.velocity * p.mass;
        total_kinetic_energy += 0.5f * p.mass * p.velocity.squaredNorm();

        float vel_mag = p.velocity.norm();
        float pos_mag = p.position.norm();

        if (vel_mag > max_velocity) max_velocity = vel_mag;
        if (pos_mag > max_position) max_position = pos_mag;
    }

    std::cout << "[Diagnostics] "
              << "Total KE: " << total_kinetic_energy << " | "
              << "Momentum: (" << total_momentum.x << ", " << total_momentum.y << ", " << total_momentum.z << ") | "
              << "Max Vel: " << max_velocity << " | "
              << "Max Pos: " << max_position << '\n';
}
