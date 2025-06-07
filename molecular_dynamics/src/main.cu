#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <sstream>
#include <cuda_runtime.h>
#include <vector>

#include "particle.cuh"
#include "../input/cli.cuh"
#include "benchmark.hpp"
#include "binning.cuh"

std::string get_timestamp_string() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm* timeinfo = std::localtime(&now_c);
    std::ostringstream oss;
    oss << std::put_time(timeinfo, "%Y%m%d_%H%M%S");
    return oss.str();
}

void generate_stable_test(Particle*& particles, int& num_particles, float sigma) {
    num_particles = 2;
    particles = new Particle[2];

    float r_min = std::pow(2.0f, 1.0f / 6.0f) * sigma;  // exact equilibrium

    particles[0].position = Vector3(0.0f, 0.0f, 0.0f);
    particles[1].position = Vector3(r_min, 0.0f, 0.0f);

    for (int i = 0; i < 2; ++i) {
        particles[i].velocity = Vector3(0.0f, 0.0f, 0.0f);
        particles[i].force = Vector3(0.0f, 0.0f, 0.0f);
        particles[i].mass = 5.0f;
    }
}

void generate_repulsive_test(Particle*& particles, int& num_particles, float sigma) {
    num_particles = 2;
    particles = new Particle[2];

    float r_min = std::pow(2.0f, 1.0f / 6.0f) * sigma; 

    particles[0].position = Vector3(0.0f, 0.0f, 0.0f);
    particles[1].position = Vector3(r_min - 0.10f, 0.0f, 0.0f); // closer than r_min

    for (int i = 0; i < 2; ++i) {
        particles[i].velocity = Vector3(0.0f, 0.0f, 0.0f);
        particles[i].mass = 5.0f;
    }
}

void generate_attractive_test(Particle*& particles, int& num_particles, float sigma) {
    num_particles = 2;
    particles = new Particle[2];
    
    float r_min = std::pow(2.0f, 1.0f / 6.0f) * sigma; 

    particles[0].position = Vector3(0.0f, 0.0f, 0.0f);
    particles[1].position = Vector3(r_min + 0.10f, 0.0f, 0.0f); // further than r_min

    for (int i = 0; i < 2; ++i) {
        particles[i].velocity = Vector3(0.0f, 0.0f, 0.0f);
        particles[i].mass = 10.0f;
    }
}

int main(int argc, char** argv) {
    SimulationConfig config;
    parse_command_line_args(argc, argv, config);

    // Extract config values for easier access
    float sigma      = config.sigma;
    float epsilon    = config.epsilon;
    float box_size[3] = {config.box_size[0], config.box_size[1], config.box_size[2]};
    float dt         = config.dt;
    int   num_steps  = config.num_steps;
    int   output_freq= config.output_freq;
    std::string output_dir = config.output_dir;
    MethodType method      = config.method;
    float rcut             = (method == MethodType::CUTOFF) ? config.rcut : 0.0f;

    // Load particle data from input file
    int num_particles = 0;
    Particle* particles;
    
    if (config.test_case == TestCaseType::STABLE) 
    {
        generate_stable_test(particles, num_particles, sigma);
        std::cout << "[INFO] Running STABLE test case\n";
    } 
    else if (config.test_case == TestCaseType::REPULSIVE) 
    {
        generate_repulsive_test(particles, num_particles, sigma);
        std::cout << "[INFO] Running REPULSIVE test case\n";
    } 
    else if (config.test_case == TestCaseType::ATTRACTIVE) 
    {
        generate_attractive_test(particles, num_particles, sigma);
        std::cout << "[INFO] Running ATTRACTIVE test case\n";
    } 
    else 
    {
        load_particles_from_file(config.input_file, particles, num_particles);
        if (!particles || num_particles == 0) {
            std::cerr << "Failed to load particles from file: " << config.input_file << "\n";
            return 1;
        }
    }

    std::ofstream csv_file;
    bool log_csv = false;
    std::string csv_path;

    // Check for --benchmark flag
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--benchmark") {
            log_csv = true;
            break;
        }
    }

    if (log_csv) {
        std::string timestamp = get_timestamp_string();
        csv_path = output_dir + "/benchmark_" + std::to_string(num_particles) + "_" + timestamp + ".csv";
        csv_file.open(csv_path);
        csv_file << "step,time_ms,num_particles\n";
    }

    // CUDA events for benchmarking
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "========== Starting Configuration ==========\n";
    print_particles(particles, num_particles);
    std::cout << "========== Simulation Configuration ==========\n";

    // Initial force computation (step 0)
    run_simulation(particles, num_particles, 0.0f, sigma, epsilon, rcut, box_size);

    // Main simulation loop with proper timing
    for (int step = 0; step < num_steps; ++step) {
        // Start timing for this step
        if (log_csv) {
            cudaEventRecord(start);
        }

        // Run the actual simulation step
        run_simulation(particles, num_particles, dt, sigma, epsilon, rcut, box_size);

        // End timing for this step
        if (log_csv) {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float step_time_ms = 0.0f;
            cudaEventElapsedTime(&step_time_ms, start, stop);
            csv_file << step << "," << step_time_ms << "," << num_particles << "\n";
        }

        print_diagnostics(particles, num_particles);
        std::cout << "Current Step: " << step << std::endl;
        print_particles(particles, num_particles);
        if ((step + 1) % output_freq == 0) write_vtk(particles, num_particles, step, output_dir); 
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (log_csv) {
        csv_file.close();
        std::cout << "Benchmark logged to: " << csv_path << "\n";

        // Generate performance plot
        std::string command = "python3 src/plot_benchmark.py \"" + csv_path + "\" --output_dir \"" + output_dir + "/plots\"";        
        int plot_status = std::system(command.c_str());
        if (plot_status != 0) {
            std::cerr << "Warning: Plot generation failed. Is Python + matplotlib installed?\n";
        }
    }

    // Final net force check
    Vector3 net_force(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < num_particles; ++i) {
        net_force += particles[i].force;
    }
    std::cout << "Net Force: (" << net_force.x << ", "
              << net_force.y << ", " << net_force.z << ")\n";

    delete[] particles;
    cleanup_simulation();     
    return 0;
}