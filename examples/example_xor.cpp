// Small end-to-end example: train XOR, save the weights, load them into a new network.
#include "nnlib/loss.h"
#include "nnlib/network.h"

int main() {
    // XOR: output is 1 when the two inputs differ
    Matrix x(4, 2), y(4, 1);
    x.at(0, 0) = 0; x.at(0, 1) = 0;  y.at(0, 0) = 0;
    x.at(1, 0) = 0; x.at(1, 1) = 1;  y.at(1, 0) = 1;
    x.at(2, 0) = 1; x.at(2, 1) = 0;  y.at(2, 0) = 1;
    x.at(3, 0) = 1; x.at(3, 1) = 1;  y.at(3, 0) = 0;

    Network net(5);
    net.add(2, 8, Act::Tanh);
    net.add(8, 1, Act::None);
    Optimizer opt(Opt::Adam, 0.01f);

    for (int epoch = 0; epoch <= 2000; epoch++) {
        Matrix out = net.forward(x);
        float loss = mse_loss(out, y);
        Matrix grad = mse_grad(out, y);
        net.backward(grad);
        net.step(opt);
        if (epoch % 500 == 0) std::cout << "epoch " << epoch << "  loss " << loss << "\n";
    }

    std::cout << "predictions (want 0 1 1 0):\n";
    net.forward(x).print();
    net.save("xor.bin");

    // a brand new network with different random weights, then load the saved ones
    Network net2(99);
    net2.add(2, 8, Act::Tanh);
    net2.add(8, 1, Act::None);
    std::cout << "before load (random weights):\n";
    net2.forward(x).print();
    net2.load("xor.bin");
    std::cout << "after load (should match the trained network):\n";
    net2.forward(x).print();
}
