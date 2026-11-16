// Does the whole thing learn? Train XOR with each optimizer and check the loss goes to ~0.
#include <cstdio>
#include "check.h"
#include "../include/nnlib/loss.h"
#include "../include/nnlib/network.h"

int main() {
    Matrix x(4, 2), y(4, 1);
    float xs[] = {0, 0, 0, 1, 1, 0, 1, 1};
    float ys[] = {0, 1, 1, 0};
    for (int i = 0; i < 8; i++) x.data[i] = xs[i];
    for (int i = 0; i < 4; i++) y.data[i] = ys[i];

    Opt kinds[] = {Opt::SGD, Opt::Momentum, Opt::Adam};
    float rates[] = {0.05f, 0.02f, 0.01f};
    const char* names[] = {"SGD", "Momentum", "Adam"};

    for (int o = 0; o < 3; o++) {
        Network net(5);
        net.add(2, 8, Act::Tanh);
        net.add(8, 1, Act::None);
        Optimizer opt(kinds[o], rates[o]);
        float loss = 1e9f;
        for (int epoch = 0; epoch < 3000; epoch++) {
            Matrix out = net.forward(x);
            loss = mse_loss(out, y);
            Matrix g = mse_grad(out, y);
            net.backward(g);
            net.step(opt);
        }
        std::printf("XOR %-8s final loss %g\n", names[o], loss);
        CHECK(loss < 1e-3f);
    }
    std::puts("test_training: all passed");
}
