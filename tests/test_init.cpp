#include "check.h"
#include "../include/nnlib/nnlib.h"


void stats(Matrix& m, const char* name, float expect_sd) {
    double s = 0, s2 = 0, mx = 0;
    for (float v : m.data) { s += v; s2 += v * v; if (std::fabs(v) > mx) mx = std::fabs(v); }
    double mean = s / m.data.size(), sd = std::sqrt(s2 / m.data.size() - mean * mean);
    std::cout << name << ": mean " << mean << "  std " << sd << " (want " << expect_sd << ")  biggest " << mx << "\n";
}
int main() {
    std::mt19937 gen(1);
    Matrix a(200, 100), b(200, 100);
    he_init(a, gen);     stats(a, "He    ", std::sqrt(2.0 / 200));
    xavier_init(b, gen); stats(b, "Xavier", std::sqrt(2.0 / 300));
    std::cout << "Xavier limit: " << std::sqrt(6.0 / 300) << "\n";
}