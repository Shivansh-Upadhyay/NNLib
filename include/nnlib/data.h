#pragma once
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include "matrix.h"

// MNIST files store each 4-byte number with the biggest byte first. This reads one of them.
inline int read_int(std::ifstream& f) {
    unsigned char b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
}

// Images -> matrix with one row per image: N x 784, pixels scaled from 0..255 to 0..1
inline Matrix load_images(const std::string& path, int max_n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cout << "can't open " << path << "\n"; std::exit(1); }

    int magic = read_int(f);
    int n = read_int(f);
    int rows = read_int(f);
    int cols = read_int(f);
    if (magic != 2051) { std::cout << path << " is not an images file\n"; std::exit(1); }

    if (n > max_n) n = max_n;
    Matrix x(n, rows * cols);
    for (float& v : x.data) {
        unsigned char pixel;
        f.read(reinterpret_cast<char*>(&pixel), 1);
        v = pixel / 255.0f;
    }
    return x;
}

// Labels -> one-hot matrix: N x 10. Digit 3 becomes [0 0 0 1 0 0 0 0 0 0]
inline Matrix load_labels(const std::string& path, int max_n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cout << "can't open " << path << "\n"; std::exit(1); }

    int magic = read_int(f);
    int n = read_int(f);
    if (magic != 2049) { std::cout << path << " is not a labels file\n"; std::exit(1); }

    if (n > max_n) n = max_n;
    Matrix y(n, 10);
    for (int i = 0; i < n; i++) {
        unsigned char label;
        f.read(reinterpret_cast<char*>(&label), 1);
        y.at(i, label) = 1;
    }
    return y;
}