#include <cstdio>
#include <random>
#include <vector>
#include "check.h"
#include "../include/nnlib/loss.h"
#include "../include/nnlib/network.h"

static Matrix make(int r, int c, std::vector<float> v) {
    Matrix m(r, c);
    m.data = v;
    return m;
}

static bool close(Matrix& a, Matrix& b, float tol = 1e-5f) {
    if (a.rows != b.rows || a.cols != b.cols) return false;
    for (size_t i = 0; i < a.data.size(); i++)
        if (std::fabs(a.data[i] - b.data[i]) > tol) return false;
    return true;
}

static float total(Matrix& m) {
    float s = 0;
    for (float v : m.data) s += v;
    return s;
}

int main() {
    // ---- activations: known values ----
    CHECK(activate(Act::ReLU, -2) == 0 && activate(Act::ReLU, 3) == 3);
    CHECK_NEAR(activate(Act::Sigmoid, 0), 0.5f, 1e-6f);
    CHECK_NEAR(activate(Act::Tanh, 1), std::tanh(1.0f), 1e-6f);
    CHECK(activate(Act::Sigmoid, -1000) == 0 && activate(Act::Sigmoid, 1000) == 1);   // no NaN

    // ---- Dense forward, checked by hand (ReLU) ----
    Dense d(2, 3, Act::ReLU, 1);
    d.W = make(2, 3, {1, -1, 0.5f, 2, 0, -1});
    d.b = make(1, 3, {0.5f, -10, 0});
    Matrix x = make(2, 2, {1, 1, 0, 2});
    Matrix y = d.forward(x);
    Matrix want_z = make(2, 3, {3.5f, -11, -0.5f, 4.5f, -10, -2});
    Matrix want_y = make(2, 3, {3.5f, 0, 0, 4.5f, 0, 0});
    CHECK(close(d.z, want_z) && close(y, want_y));

    // ---- initialization statistics ----
    std::mt19937 gen(1);
    Matrix w(200, 100);
    he_init(w, gen);
    double s = 0, s2 = 0;
    for (float v : w.data) { s += v; s2 += double(v) * v; }
    double mean = s / w.data.size(), sd = std::sqrt(s2 / w.data.size() - mean * mean);
    CHECK(std::fabs(mean) < 0.01 && std::fabs(sd - std::sqrt(2.0 / 200)) < 0.05 * std::sqrt(2.0 / 200));
    xavier_init(w, gen);
    double limit = std::sqrt(6.0 / 300);
    for (float v : w.data) CHECK(std::fabs(v) <= limit);

    // ---- gradient check: backward() against "nudge and see" ----
    // loss = sum of all outputs, so d_out is all ones.
    Act acts[] = {Act::None, Act::ReLU, Act::Sigmoid, Act::Tanh};
    for (Act a : acts) {
        Dense l(3, 2, a, 3);
        Matrix in = make(2, 3, {0.5f, -1, 2, 1, 0.3f, -0.7f});
        Matrix out = l.forward(in);
        Matrix ones(2, 2);
        for (float& v : ones.data) v = 1;
        Matrix dx = l.backward(ones);               // fills l.dW, l.db and returns dX
        Matrix dW = l.dW, db = l.db;
        const float h = 0.01f;

        // nudge number k of m, return (loss(up) - loss(down)) / 2h
        auto numeric = [&](Matrix& m, size_t k) {
            float keep = m.data[k];
            m.data[k] = keep + h; Matrix o1 = l.forward(in); float up = total(o1);
            m.data[k] = keep - h; Matrix o2 = l.forward(in); float down = total(o2);
            m.data[k] = keep;
            return (up - down) / (2 * h);
        };
        for (size_t k = 0; k < l.W.data.size(); k++) CHECK_NEAR(numeric(l.W, k), dW.data[k], 1e-3f);
        for (size_t k = 0; k < l.b.data.size(); k++) CHECK_NEAR(numeric(l.b, k), db.data[k], 1e-3f);
        for (size_t k = 0; k < in.data.size(); k++) CHECK_NEAR(numeric(in, k), dx.data[k], 1e-3f);
    }

    // ---- gradient check through a whole 2-layer Network (also tests skipping dX in layer 0) ----
    {
        Network net(7);
        net.add(3, 4, Act::Tanh);
        net.add(4, 2, Act::None);
        Matrix in = make(2, 3, {0.5f, -1, 2, 1, 0.3f, -0.7f});
        Matrix target = make(2, 2, {1, 0, 0, 1});
        Matrix out = net.forward(in);
        Matrix g = mse_grad(out, target);
        net.backward(g);
        Matrix dW0 = net.layers[0].dW;
        const float h = 0.01f;
        for (size_t k = 0; k < dW0.data.size(); k++) {
            float keep = net.layers[0].W.data[k];
            net.layers[0].W.data[k] = keep + h; Matrix o1 = net.forward(in); float up = mse_loss(o1, target);
            net.layers[0].W.data[k] = keep - h; Matrix o2 = net.forward(in); float down = mse_loss(o2, target);
            net.layers[0].W.data[k] = keep;
            CHECK_NEAR((up - down) / (2 * h), dW0.data[k], 1e-3f);
        }
    }

    // ---- save and load: same predictions ----
    {
        Network a(1);
        a.add(3, 5, Act::ReLU);
        a.add(5, 2, Act::None);
        Network b(99);                              // different random weights
        b.add(3, 5, Act::ReLU);
        b.add(5, 2, Act::None);
        Matrix in = make(2, 3, {0.5f, -1, 2, 1, 0.3f, -0.7f});
        a.save("test_net.bin");
        b.load("test_net.bin");
        Matrix ya = a.forward(in), yb = b.forward(in);
        CHECK(close(ya, yb, 0));                    // exactly equal
        std::remove("test_net.bin");
    }

    std::puts("test_layers: all passed");
}
