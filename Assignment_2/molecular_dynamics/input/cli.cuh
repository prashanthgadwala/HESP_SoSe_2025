#pragma once
#include <string>

enum class TestCaseType { NONE, STABLE, REPULSIVE, ATTRACTIVE };

struct SimulationConfig {
    std::string input_file;
    std::string output_dir;
    float dt;
    int num_steps;
    float sigma;
    float epsilon;
    int output_freq;
    bool benchmark;
    TestCaseType test_case = TestCaseType::NONE;
};

void parse_command_line_args(int argc, char** argv, SimulationConfig& config);