#pragma once
#include <cmath>
#include "matrix.h"

// loss = average of (pred - target)^2 over all numbers
inline float mse_loss(Matrix& pred, Matrix& target) {
    float sum = 0;
    for (size_t i = 0; i < pred.data.size(); i++) {
        float d = pred.data[i] - target.data[i];
        sum += d * d;
    }
    return sum / pred.data.size();
}

// gradient of the loss with respect to pred: 2 * (pred - target) / count
inline Matrix mse_grad(Matrix& pred, Matrix& target) {
    Matrix g(pred.rows, pred.cols);
    for (size_t i = 0; i < pred.data.size(); i++)
        g.data[i] = 2 * (pred.data[i] - target.data[i]) / pred.data.size();
    return g;
}

// softmax turns each row of scores into probabilities that add up to 1
inline Matrix softmax(Matrix& z) {
    Matrix p(z.rows, z.cols);
    for (int i = 0; i < z.rows; i++) {
        float mx = z.at(i, 0);
        for (int j = 1; j < z.cols; j++)
            if (z.at(i, j) > mx) mx = z.at(i, j);       // biggest score in this row

        float sum = 0;
        for (int j = 0; j < z.cols; j++) {
            p.at(i, j) = std::exp(z.at(i, j) - mx);     // subtract max so exp() never overflows
            sum += p.at(i, j);
        }
        for (int j = 0; j < z.cols; j++) p.at(i, j) /= sum;
    }
    return p;
}

// y is one-hot: for each row, 1 at the correct class and 0 elsewhere
// loss = average of -log(probability given to the correct class)
inline float cross_entropy_loss(Matrix& z, Matrix& y) {
    Matrix p = softmax(z);
    float sum = 0;
    for (size_t i = 0; i < p.data.size(); i++)
        sum -= y.data[i] * std::log(p.data[i] + 1e-12f);   // 1e-12 avoids log(0)
    return sum / z.rows;
}

// gradient with respect to the scores z: (softmax(z) - y) / number_of_samples
inline Matrix cross_entropy_grad(Matrix& z, Matrix& y) {
    Matrix p = softmax(z);
    Matrix g(z.rows, z.cols);
    for (size_t i = 0; i < p.data.size(); i++)
        g.data[i] = (p.data[i] - y.data[i]) / z.rows;
    return g;
}