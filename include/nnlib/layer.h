#pragma once
#include <cmath>
#include <random>
#include "init.h"
#include "matrix.h"

enum class Act
{
    None,
    ReLU,
    Sigmoid,
    Tanh
};

// activation applied to one number
inline float activate(Act a, float z)
{
    if (a == Act::ReLU)
        return z > 0 ? z : 0;
    if (a == Act::Sigmoid)
        return 1.0f / (1.0f + std::exp(-z));
    if (a == Act::Tanh)
        return std::tanh(z);
    return z; 
}

// slope of the activation at z (needed for backward)
inline float activate_grad(Act a, float z)
{
    if (a == Act::ReLU)
        return z > 0 ? 1.0f : 0.0f;
    if (a == Act::Sigmoid)
    {
        float s = activate(Act::Sigmoid, z);
        return s * (1 - s);
    }
    if (a == Act::Tanh)
    {
        float t = std::tanh(z);
        return 1 - t * t;
    }
    return 1.0f; 
}

// One layer of the network:  output = activation(X * W + b)
class Dense
{
public:
    Matrix W, b; // weights, bias 
    Act act;
    Matrix input, z; // saved by forward, used by backward
    Matrix dW, db;   // gradients filled by backward

    Dense(int in, int out, Act act, int seed = 42)
        : W(in, out), b(1, out), act(act), input(0, 0), z(0, 0), dW(in, out), db(1, out)
    {
        std::mt19937 gen(seed); // same seed = same weights every run
        if (act == Act::ReLU)
            he_init(W, gen); // ReLU -> He
        else
            xavier_init(W, gen); // everything else -> Xavier. b stays 0
    }

    Matrix forward(Matrix &x)
    {
        input = x; // save for backward
        Matrix xw = matmul(x, W);
        z = add_bias(xw, b); // z = X*W + b

        Matrix out = z;
        for (float &v : out.data)
            v = activate(act, v);
        return out;
    }

    // d_out = how much the loss changes when this layer's output changes
    // returns dX = same thing for this layer's input (to pass to the layer before)
    Matrix backward(Matrix &d_out, bool need_dx = true)
    {
        Matrix dz = d_out;
        for (size_t i = 0; i < dz.data.size(); i++)
            dz.data[i] *= activate_grad(act, z.data[i]); // chain rule through the activation

        Matrix xt = transpose(input);
        dW = matmul(xt, dz); // dW = Xt * dz
        db = sum_rows(dz);   // db = sum of dz over the batch
        if (!need_dx)
            return Matrix(0, 0); // first layer: nobody needs dX, skip the work
        Matrix wt = transpose(W);
        return matmul(dz, wt); // dX = dz * Wt
    }
};