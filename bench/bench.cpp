// Benchmark: training time per epoch + batch-1 inference latency.
// Same network as the PyTorch script: 784 -> 128 (ReLU) -> 10, SGD lr 0.1, batch 64.
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "../include/nnlib/data.h"
#include "../include/nnlib/loss.h"
#include "../include/nnlib/network.h"

using Clock = std::chrono::steady_clock;

Matrix get_rows(Matrix& m, std::vector<int>& idx, int start, int count) {
    Matrix out(count, m.cols);
    for (int i = 0; i < count; i++)
        for (int j = 0; j < m.cols; j++)
            out.at(i, j) = m.at(idx[start + i], j);
    return out;
}

int argmax_row(Matrix& m, int i) {
    int best = 0;
    for (int j = 1; j < m.cols; j++)
        if (m.at(i, j) > m.at(i, best)) best = j;
    return best;
}

float accuracy(Network& net, Matrix& x, Matrix& y) {
    Matrix scores = net.forward(x);
    int correct = 0;
    for (int i = 0; i < x.rows; i++)
        if (argmax_row(scores, i) == argmax_row(y, i)) correct++;
    return (float)correct / x.rows;
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "data";
    Matrix train_x = load_images(dir + "/train-images-idx3-ubyte", 60000);
    Matrix train_y = load_labels(dir + "/train-labels-idx1-ubyte", 60000);
    Matrix test_x = load_images(dir + "/t10k-images-idx3-ubyte", 10000);
    Matrix test_y = load_labels(dir + "/t10k-labels-idx1-ubyte", 10000);

    Network net(1);
    net.add(784, 128, Act::ReLU);
    net.add(128, 10, Act::None);
    Optimizer opt(Opt::SGD, 0.1f);

    const int epochs = 5, batch = 64;
    std::mt19937 gen(0);
    std::vector<int> order(train_x.rows);
    std::iota(order.begin(), order.end(), 0);

    std::cout << std::fixed << std::setprecision(4);

    // ---- 1. training speed ----
    for (int epoch = 1; epoch <= epochs; epoch++) {
        std::shuffle(order.begin(), order.end(), gen);
        auto t0 = Clock::now();

        float total_loss = 0;
        int batches = 0;
        for (int start = 0; start + batch <= train_x.rows; start += batch) {
            Matrix x = get_rows(train_x, order, start, batch);
            Matrix y = get_rows(train_y, order, start, batch);
            Matrix scores = net.forward(x);
            total_loss += cross_entropy_loss(scores, y);
            batches++;
            Matrix d_scores = cross_entropy_grad(scores, y);
            net.backward(d_scores);
            net.step(opt);
        }
        double secs = std::chrono::duration<double>(Clock::now() - t0).count();
        std::cout << "nnlib   epoch " << epoch << "  time " << secs << " s  loss " << total_loss / batches
                  << "  test acc " << accuracy(net, test_x, test_y) * 100 << "%\n";
    }

    // ---- 2. batch-1 inference latency ----
    std::vector<int> ids(1000);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<Matrix> singles;                       // 1000 single-image inputs, prepared in advance
    for (int i = 0; i < 1000; i++) singles.push_back(get_rows(test_x, ids, i, 1));

    for (int i = 0; i < 200; i++) net.forward(singles[i % 1000]);        // warm-up

    const int runs = 10000;
    std::vector<double> us(runs);
    for (int i = 0; i < runs; i++) {
        auto t0 = Clock::now();
        Matrix out = net.forward(singles[i % 1000]);
        us[i] = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
    }
    std::sort(us.begin(), us.end());
    std::cout << "nnlib   batch-1 latency:  p50 " << us[runs / 2] << " us   p99 " << us[runs * 99 / 100] << " us\n";
}
