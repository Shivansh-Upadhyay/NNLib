#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "nnlib/data.h"
#include "nnlib/loss.h"
#include "nnlib/network.h"

// copy `count` rows of m, starting at position `start` in the list idx
Matrix get_rows(Matrix& m, std::vector<int>& idx, int start, int count) {
    Matrix out(count, m.cols);
    for (int i = 0; i < count; i++)
        for (int j = 0; j < m.cols; j++)
            out.at(i, j) = m.at(idx[start + i], j);
    return out;
}

// index of the biggest number in row i
int argmax_row(Matrix& m, int i) {
    int best = 0;
    for (int j = 1; j < m.cols; j++)
        if (m.at(i, j) > m.at(i, best)) best = j;
    return best;
}

// fraction of images where the biggest score is the correct digit
float accuracy(Network& net, Matrix& x, Matrix& y) {
    Matrix scores = net.forward(x);
    int correct = 0;
    for (int i = 0; i < x.rows; i++)
        if (argmax_row(scores, i) == argmax_row(y, i)) correct++;
    return (float)correct / x.rows;
}

int main(int argc, char** argv) {
    std::string dir = argc > 2 ? argv[2] : "data";       // usage: train_mnist [sgd|momentum|adam] [data folder]
    Matrix train_x = load_images(dir + "/train-images-idx3-ubyte", 60000);
    Matrix train_y = load_labels(dir + "/train-labels-idx1-ubyte", 60000);
    Matrix test_x = load_images(dir + "/t10k-images-idx3-ubyte", 10000);
    Matrix test_y = load_labels(dir + "/t10k-labels-idx1-ubyte", 10000);
    std::cout << "loaded " << train_x.rows << " train, " << test_x.rows << " test images\n";

    Network net(1);
    net.add(784, 128, Act::ReLU);     // 784 pixels -> 128 hidden (ReLU)
    net.add(128, 10, Act::None);      // 128 hidden -> 10 scores (one per digit)

    // choose optimizer from the command line:  ./train sgd | momentum | adam
    std::string which = argc > 1 ? argv[1] : "adam";
    Optimizer opt = which == "sgd"      ? Optimizer(Opt::SGD, 0.1f)
                  : which == "momentum" ? Optimizer(Opt::Momentum, 0.02f)
                                        : Optimizer(Opt::Adam, 0.001f);
    std::cout << "optimizer: " << which << "\n";

    const int epochs = 5;
    const int batch = 64;
    std::mt19937 gen(0);
    std::vector<int> order(train_x.rows);
    std::iota(order.begin(), order.end(), 0);

    for (int epoch = 1; epoch <= epochs; epoch++) {
        std::shuffle(order.begin(), order.end(), gen);

        float total_loss = 0;
        int batches = 0;
        for (int start = 0; start + batch <= train_x.rows; start += batch) {
            Matrix x = get_rows(train_x, order, start, batch);
            Matrix y = get_rows(train_y, order, start, batch);

            Matrix scores = net.forward(x);                    // forward
            total_loss += cross_entropy_loss(scores, y);
            batches++;

            Matrix d_scores = cross_entropy_grad(scores, y);   // backward
            net.backward(d_scores);

            net.step(opt);                                     // update
        }
        std::cout << "epoch " << epoch << "  loss " << total_loss / batches
                  << "  test accuracy " << accuracy(net, test_x, test_y) * 100 << "%\n";
    }

    net.save("mnist.bin");
    std::cout << "saved weights to mnist.bin\n";
}
