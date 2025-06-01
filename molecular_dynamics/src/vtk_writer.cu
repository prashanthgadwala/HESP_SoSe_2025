#include <fstream>
#include <iomanip>
#include <filesystem>
#include <string>
#include "particle.cuh"

__host__ void write_vtk(const Particle* particles, int num_particles, int step, const std::string& output_dir) {
    std::ostringstream filename;
    filename << output_dir << "/particles_" 
             << std::setw(4) << std::setfill('0') << step << ".vtk";

    std::ofstream vtk_file(filename.str());
    if (!vtk_file.is_open()) {
        std::cerr << "Failed to write VTK file: " << filename.str() << "\n";
        return;
    }

    // VTK header
    vtk_file << "# vtk DataFile Version 3.0\n";
    vtk_file << "Molecular Dynamics Particle Data\n";
    vtk_file << "ASCII\n";
    vtk_file << "DATASET POLYDATA\n";
    vtk_file << "POINTS " << num_particles << " float\n";

    // Positions
    for (int i = 0; i < num_particles; ++i) {
        const auto& p = particles[i];
        vtk_file << p.position.x << " " << p.position.y << " " << p.position.z << "\n";
    }

    // Vertices
    vtk_file << "VERTICES " << num_particles << " " << num_particles * 2 << "\n";
    for (int i = 0; i < num_particles; ++i) {
        vtk_file << "1 " << i << "\n";
    }

    // Point data
    vtk_file << "POINT_DATA " << num_particles << "\n";
    
    // Velocities
    vtk_file << "VECTORS velocity float\n";
    for (int i = 0; i < num_particles; ++i) {
        const auto& v = particles[i].velocity;
        vtk_file << v.x << " " << v.y << " " << v.z << "\n";
    }
    
    // Forces
    vtk_file << "VECTORS force float\n";
    for (int i = 0; i < num_particles; ++i) {
        const auto& f = particles[i].force;
        vtk_file << f.x << " " << f.y << " " << f.z << "\n";
    }
    
    // Mass
    vtk_file << "SCALARS mass float 1\n";
    vtk_file << "LOOKUP_TABLE default\n";
    for (int i = 0; i < num_particles; ++i) {
        vtk_file << particles[i].mass << "\n";
    }

    vtk_file.close();
}