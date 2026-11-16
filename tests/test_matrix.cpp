#include <random>
#include <vector>
#include "check.h"
#include "../include/nnlib/matrix.h"

// build a matrix from a list of numbers (row by row)
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

// the plain 3-loop version, used as the reference answer
static Matrix naive_matmul(Matrix& a, Matrix& b) {
    Matrix c(a.rows, b.cols);
    for (int i = 0; i < a.rows; i++)
        for (int j = 0; j < b.cols; j++) {
            float s = 0;
            for (int k = 0; k < a.cols; k++) s += a.at(i, k) * b.at(k, j);
            c.at(i, j) = s;
        }
    return c;
}

int main() {
    // matmul, checked by hand
    Matrix a = make(2, 3, {1, 2, 3, 4, 5, 6});
    Matrix b = make(3, 2, {7, 8, 9, 10, 11, 12});
    Matrix want = make(2, 2, {58, 64, 139, 154});
    Matrix got = matmul(a, b);
    CHECK(close(got, want));

    // blocked matmul must match the plain version, including odd sizes
    std::mt19937 gen(1);
    std::uniform_real_distribution<float> dist(-1, 1);
    int shapes[][3] = {{1, 1, 1}, {2, 3, 2}, {5, 7, 3}, {4, 128, 10}, {64, 784, 128},
                       {784, 64, 128}, {7, 300, 13}, {1, 784, 128}, {3, 129, 5}};
    for (auto& s : shapes) {
        Matrix x(s[0], s[1]), y(s[1], s[2]);
        for (float& v : x.data) v = dist(gen);
        for (float& v : y.data) v = dist(gen);
        Matrix fast = matmul(x, y), slow = naive_matmul(x, y);
        CHECK(close(fast, slow, 1e-4f));
    }

    // transpose: shape, values, and (AB)^T = B^T A^T
    Matrix t = transpose(a);
    CHECK(t.rows == 3 && t.cols == 2 && t.at(2, 1) == 6 && t.at(0, 1) == 4);
    Matrix lhs = transpose(got);
    Matrix bt = transpose(b), at = transpose(a);
    Matrix rhs = matmul(bt, at);
    CHECK(close(lhs, rhs));

    // element-wise operations
    Matrix p = make(2, 2, {1, 2, 3, 4}), q = make(2, 2, {5, 6, 7, 8});
    Matrix r1 = add(p, q), r2 = sub(q, p), r3 = mul(p, q), r4 = scale(p, 2);
    Matrix e1 = make(2, 2, {6, 8, 10, 12}), e2 = make(2, 2, {4, 4, 4, 4});
    Matrix e3 = make(2, 2, {5, 12, 21, 32}), e4 = make(2, 2, {2, 4, 6, 8});
    CHECK(close(r1, e1) && close(r2, e2) && close(r3, e3) && close(r4, e4));

    // bias add and sum over the batch
    Matrix bias = make(1, 3, {10, 20, 30});
    Matrix ab = add_bias(a, bias), eab = make(2, 3, {11, 22, 33, 14, 25, 36});
    CHECK(close(ab, eab));
    Matrix sr = sum_rows(a), esr = make(1, 3, {5, 7, 9});
    CHECK(close(sr, esr));

    // copy is independent of the original
    Matrix c = a;
    c.at(0, 0) = 99;
    CHECK(a.at(0, 0) == 1);

    std::puts("test_matrix: all passed");
}
