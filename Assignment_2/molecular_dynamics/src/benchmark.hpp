#ifndef BENCHMARK_HPP
#define BENCHMARK_HPP

#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iostream>

class BenchmarkLogger {
private:
    std::ofstream file;
    std::vector<double> step_times;
    std::chrono::high_resolution_clock::time_point last_time;
    bool enabled;
    int num_particles;

public:
    BenchmarkLogger(bool enable_logging, int particle_count = 0, const std::string& output_dir = "output") 
        : enabled(enable_logging), num_particles(particle_count) {
        if (enabled) {
            auto now = std::chrono::system_clock::now();
            auto now_c = std::chrono::system_clock::to_time_t(now);
            std::ostringstream oss;
            oss << output_dir << "/benchmark_" << num_particles << "_"
                << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S")
                << ".csv";
            file.open(oss.str());
            file << "step,time_ms,num_particles\n";
        }
        last_time = std::chrono::high_resolution_clock::now();
    }

    void start_step() {
        if (!enabled) return;
        last_time = std::chrono::high_resolution_clock::now();
    }

    void end_step(int step) {
        if (!enabled) return;
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(now - last_time).count();
        step_times.push_back(elapsed_ms);
        file << step << "," << elapsed_ms << "," << num_particles << "\n";
        file.flush(); // Ensure data is written immediately
    }

    void log_step(int step) {
        if (!enabled) return;
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(now - last_time).count();
        step_times.push_back(elapsed_ms);
        file << step << "," << elapsed_ms << "," << num_particles << "\n";
        last_time = now;
    }

    // Get performance statistics
    struct PerformanceStats {
        double mean_ms;
        double std_dev_ms;
        double min_ms;
        double max_ms;
        double median_ms;
        size_t num_samples;
    };

    PerformanceStats get_stats() const {
        PerformanceStats stats = {0.0, 0.0, 0.0, 0.0, 0.0, 0};
        
        if (step_times.empty()) return stats;
        
        stats.num_samples = step_times.size();
        
        // Mean
        stats.mean_ms = std::accumulate(step_times.begin(), step_times.end(), 0.0) / step_times.size();
        
        // Min/Max
        auto minmax = std::minmax_element(step_times.begin(), step_times.end());
        stats.min_ms = *minmax.first;
        stats.max_ms = *minmax.second;
        
        // Standard deviation
        double sum_sq_diff = 0.0;
        for (double time : step_times) {
            double diff = time - stats.mean_ms;
            sum_sq_diff += diff * diff;
        }
        stats.std_dev_ms = std::sqrt(sum_sq_diff / step_times.size());
        
        // Median
        std::vector<double> sorted_times = step_times;
        std::sort(sorted_times.begin(), sorted_times.end());
        size_t mid = sorted_times.size() / 2;
        if (sorted_times.size() % 2 == 0) {
            stats.median_ms = (sorted_times[mid-1] + sorted_times[mid]) / 2.0;
        } else {
            stats.median_ms = sorted_times[mid];
        }
        
        return stats;
    }

    void print_summary() const {
        if (!enabled || step_times.empty()) return;
        
        auto stats = get_stats();
        std::cout << "\n========== Performance Summary ==========\n";
        std::cout << "Particles: " << num_particles << "\n";
        std::cout << "Samples: " << stats.num_samples << "\n";
        std::cout << "Mean time per step: " << std::fixed << std::setprecision(3) 
                  << stats.mean_ms << " ms\n";
        std::cout << "Standard deviation: " << stats.std_dev_ms << " ms\n";
        std::cout << "Min time: " << stats.min_ms << " ms\n";
        std::cout << "Max time: " << stats.max_ms << " ms\n";
        std::cout << "Median time: " << stats.median_ms << " ms\n";
        std::cout << "==========================================\n";
    }

    ~BenchmarkLogger() {
        if (file.is_open()) {
            file.close();
        }
        if (enabled) {
            print_summary();
        }
    }
};

#endif // BENCHMARK_HPP