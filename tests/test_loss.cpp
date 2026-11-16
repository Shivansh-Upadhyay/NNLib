#include <vector>
#include "check.h"
#include "../include/nnlib/loss.h"

static Matrix make(int r, int c, std::vector<float> v) {
    Matrix m(r, c);
    m.data = v;
    return m;
}

int main() {
    // ---- MSE ----
    Matrix p = make(1, 2, {1, 2}), t = make(1, 2, {0, 4});
    CHECK_NEAR(mse_loss(p, t), 2.5f, 1e-6f);
    Matrix g = mse_grad(p, t);
    CHECK_NEAR(g.data[0], 1.0f, 1e-6f);
    CHECK_NEAR(g.data[1], -2.0f, 1e-6f);

    // ---- softmax: rows add up to 1, equal scores give equal probabilities ----
    Matrix z = make(2, 3, {1, 2, 3, 0, 0, 0});
    Matrix sm = softmax(z);
    for (int i = 0; i < 2; i++) CHECK_NEAR(sm.at(i, 0) + sm.at(i, 1) + sm.at(i, 2), 1.0f, 1e-6f);
    CHECK_NEAR(sm.at(1, 0), 1.0f / 3, 1e-6f);

    // ---- cross-entropy ----
    Matrix z2 = make(1, 2, {0, 0}), y2 = make(1, 2, {1, 0});
    CHECK_NEAR(cross_entropy_loss(z2, y2), std::log(2.0f), 1e-5f);
    Matrix g2 = cross_entropy_grad(z2, y2);
    CHECK_NEAR(g2.data[0], -0.5f, 1e-6f);
    CHECK_NEAR(g2.data[1], 0.5f, 1e-6f);

    // huge scores must not give NaN or inf
    Matrix big = make(1, 2, {1000, 0});
    float l = cross_entropy_loss(big, y2);
    CHECK(std::isfinite(l) && l < 1e-6f);

    // ---- cross-entropy gradient against "nudge and see", 2 samples x 3 classes ----
    Matrix z3 = make(2, 3, {0.5f, -1, 2, 1, 0.3f, -0.7f});
    Matrix y3 = make(2, 3, {0, 0, 1, 1, 0, 0});
    Matrix g3 = cross_entropy_grad(z3, y3);
    const float h = 0.01f;
    for (size_t k = 0; k < z3.data.size(); k++) {
        float keep = z3.data[k];
        z3.data[k] = keep + h; float up = cross_entropy_loss(z3, y3);
        z3.data[k] = keep - h; float down = cross_entropy_loss(z3, y3);
        z3.data[k] = keep;
        CHECK_NEAR((up - down) / (2 * h), g3.data[k], 1e-3f);
    }

    std::puts("test_loss: all passed");
}
