#pragma once
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "layer.h"
#include "optimizer.h"

// A stack of Dense layers. Data flows forward through them, gradients flow backward.
class Network
{
public:
    std::vector<Dense> layers;
    int seed;

    Network(int seed = 1) : seed(seed) {}

    // add a layer to the end; each layer gets a different seed so weights differ
    void add(int in, int out, Act act)
    {
        layers.push_back(Dense(in, out, act, seed + (int)layers.size()));
    }

    Matrix forward(Matrix &x)
    {
        Matrix out = x;
        for (Dense &layer : layers)
            out = layer.forward(out); // output of one layer is the input of the next
        return out;
    }

    // d_out = gradient from the loss; goes through the layers in reverse order
    void backward(Matrix &d_out)
    {
        Matrix grad = d_out;
        for (int i = (int)layers.size() - 1; i >= 0; i--)
            grad = layers[i].backward(grad, i > 0); // layer 0 has no layer before it
    }

    void step(Optimizer &opt) { opt.step(layers); }

        // Save all weights and biases to a file.
    // Format: number of layers, then for each layer: rows, cols, W numbers, b numbers.
    // The activations (ReLU, ...) are not saved: they come from your add() calls.
    void save(const std::string& path) {
        std::ofstream f(path, std::ios::binary);
        if (!f) { std::cout << "can't write " << path << "\n"; std::exit(1); }
        int count = (int)layers.size();
        f.write(reinterpret_cast<const char*>(&count), sizeof(int));
        for (Dense& l : layers) {
            int dims[2] = {l.W.rows, l.W.cols};
            f.write(reinterpret_cast<const char*>(dims), sizeof(dims));
            f.write(reinterpret_cast<const char*>(l.W.data.data()), l.W.data.size() * sizeof(float));
            f.write(reinterpret_cast<const char*>(l.b.data.data()), l.b.data.size() * sizeof(float));
        }
    }

    // Load weights written by save(). Build the network with the same add() calls first.
    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { std::cout << "can't open " << path << "\n"; std::exit(1); }
        int count = 0;
        f.read(reinterpret_cast<char*>(&count), sizeof(int));
        if (count != (int)layers.size()) {
            std::cout << "load: file has " << count << " layers, network has " << layers.size() << "\n";
            std::exit(1);
        }
        for (Dense& l : layers) {
            int dims[2];
            f.read(reinterpret_cast<char*>(dims), sizeof(dims));
            if (dims[0] != l.W.rows || dims[1] != l.W.cols) {
                std::cout << "load: layer size in file does not match the network\n";
                std::exit(1);
            }
            f.read(reinterpret_cast<char*>(l.W.data.data()), l.W.data.size() * sizeof(float));
            f.read(reinterpret_cast<char*>(l.b.data.data()), l.b.data.size() * sizeof(float));
        }
        if (!f) { std::cout << "load: file is too short or damaged\n"; std::exit(1); }
    }
};