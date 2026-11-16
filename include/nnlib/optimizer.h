#pragma once
#include <cmath>
#include <vector>
#include "layer.h"

enum class Opt { SGD, Momentum, Adam };

class Optimizer {
public:
    Opt type;
    float lr;                              // learning rate
    float beta1 = 0.9f, beta2 = 0.999f;    // memory settings for Momentum / Adam
    float eps = 1e-8f;
    int t = 0;                             // number of steps taken so far (Adam needs it)
    std::vector<Matrix> mW, mB, vW, vB;    // per-layer memory

    Optimizer(Opt type, float lr) : type(type), lr(lr) {}

    // update every layer once (call after backward)
    void step(std::vector<Dense>& layers) {
        t++;
        for (int i = 0; i < (int)layers.size(); i++) {
            if ((int)mW.size() <= i) {     // first time: make empty memory for this layer
                Matrix& L = layers[i].W;
                mW.push_back(Matrix(L.rows, L.cols));  vW.push_back(Matrix(L.rows, L.cols));
                mB.push_back(Matrix(1, L.cols));       vB.push_back(Matrix(1, L.cols));
            }
            update(layers[i].W, layers[i].dW, mW[i], vW[i]);
            update(layers[i].b, layers[i].db, mB[i], vB[i]);
        }
    }

private:
    // p = parameters, g = their gradient, m and v = memory
    void update(Matrix& p, Matrix& g, Matrix& m, Matrix& v) {
        for (size_t k = 0; k < p.data.size(); k++) {
            float grad = g.data[k];
            if (type == Opt::SGD) {
                p.data[k] -= lr * grad;
            } else if (type == Opt::Momentum) {
                m.data[k] = beta1 * m.data[k] + grad;          // keep a running "speed"
                p.data[k] -= lr * m.data[k];
            } else {  // Adam
                m.data[k] = beta1 * m.data[k] + (1 - beta1) * grad;           // average of gradients
                v.data[k] = beta2 * v.data[k] + (1 - beta2) * grad * grad;    // average of squared gradients
                float m_hat = m.data[k] / (1 - std::pow(beta1, t));           // fix the start-up bias
                float v_hat = v.data[k] / (1 - std::pow(beta2, t));
                p.data[k] -= lr * m_hat / (std::sqrt(v_hat) + eps);
            }
        }
    }
};