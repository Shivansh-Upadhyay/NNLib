#pragma once
#include <cmath>
#include <random>
#include "matrix.h"

// He init: random numbers with mean 0 and spread sqrt(2 / fan_in). Use with ReLU.
// W is (fan_in x fan_out), so fan_in = W.rows
inline void he_init(Matrix& W, std::mt19937& gen) {
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / W.rows));
    for (float& w : W.data) w = dist(gen);
}

// Xavier init: random numbers between -limit and +limit, limit = sqrt(6 / (fan_in + fan_out)).
// Use with sigmoid, tanh, or no activation.
inline void xavier_init(Matrix& W, std::mt19937& gen) {
    float limit = std::sqrt(6.0f / (W.rows + W.cols));
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (float& w : W.data) w = dist(gen);
}