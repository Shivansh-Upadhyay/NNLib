#pragma once
#include <cstdlib>
#include <iostream>
#include <vector>

class Matrix
{
public:
    int rows, cols;
    std::vector<float> data; 

    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0f) {} 

    float &at(int i, int j) { return data[i * cols + j]; } // element at row i, column j

    void print()
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                std::cout << at(i, j) << " ";
            }
            std::cout << "\n";
        }
    }
};


inline Matrix add(Matrix &a, Matrix &b)
{
    Matrix result(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++)
        for (int j = 0; j < a.cols; j++)
            result.at(i, j) = a.at(i, j) + b.at(i, j);
    return result;
}


inline Matrix sub(Matrix &a, Matrix &b)
{
    Matrix result(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++)
        for (int j = 0; j < a.cols; j++)
            result.at(i, j) = a.at(i, j) - b.at(i, j);
    return result;
}

// element-wise multiply (same size, multiply matching positions)
inline Matrix mul(Matrix &a, Matrix &b)
{
    Matrix result(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++)
        for (int j = 0; j < a.cols; j++)
            result.at(i, j) = a.at(i, j) * b.at(i, j);
    return result;
}

// multiply every number by s
inline Matrix scale(Matrix &a, float s)
{
    Matrix result(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++)
        for (int j = 0; j < a.cols; j++)
            result.at(i, j) = a.at(i, j) * s;
    return result;
}

// matrix multiply: (n x k) * (k x m) = (n x m)
// Two speed tricks (the answer is exactly the same as the plain 3-loop version):
//   1. do 4 rows of a at once, so every number loaded from b is used 4 times
//   2. go through k in chunks of 128, so that part of b stays in the CPU cache
inline Matrix matmul(Matrix &a, Matrix &b)
{
    if (a.cols != b.rows)
    {
        std::cout << "matmul: shapes don't match\n";
        std::exit(1);
    }
    const int n = a.rows, K = a.cols, m = b.cols;
    const int KB = 128; // chunk size for k
    Matrix result(n, m);
    const float *A = a.data.data();
    const float *B = b.data.data();
    float *R = result.data.data();

    for (int k0 = 0; k0 < K; k0 += KB)
    {
        const int k1 = (k0 + KB < K) ? k0 + KB : K; // this chunk is k0 ... k1-1
        int i = 0;
        for (; i + 4 <= n; i += 4)
        { // 4 rows of the result at a time
            float *r0 = R + (i + 0) * m;
            float *r1 = R + (i + 1) * m;
            float *r2 = R + (i + 2) * m;
            float *r3 = R + (i + 3) * m;
            for (int k = k0; k < k1; k++)
            {
                const float a0 = A[(i + 0) * K + k];
                const float a1 = A[(i + 1) * K + k];
                const float a2 = A[(i + 2) * K + k];
                const float a3 = A[(i + 3) * K + k];
                const float *brow = B + k * m;
                for (int j = 0; j < m; j++)
                {
                    const float bv = brow[j]; // loaded once, used 4 times
                    r0[j] += a0 * bv;
                    r1[j] += a1 * bv;
                    r2[j] += a2 * bv;
                    r3[j] += a3 * bv;
                }
            }
        }
        for (; i < n; i++) // leftover rows (if n is not a multiple of 4)
            for (int k = k0; k < k1; k++)
            {
                const float aik = A[i * K + k];
                for (int j = 0; j < m; j++)
                    R[i * m + j] += aik * B[k * m + j];
            }
    }
    return result;
}


inline Matrix transpose(Matrix &a)
{
    Matrix result(a.cols, a.rows);
    for (int i = 0; i < a.rows; i++)
        for (int j = 0; j < a.cols; j++)
            result.at(j, i) = a.at(i, j);
    return result;
}

// add a 1-row matrix (bias) to every row of m
inline Matrix add_bias(Matrix &m, Matrix &bias)
{
    Matrix result(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++)
            result.at(i, j) = m.at(i, j) + bias.at(0, j);
    return result;
}

// add up all rows -> 1 row (used for the bias gradient)
inline Matrix sum_rows(Matrix &m)
{
    Matrix result(1, m.cols);
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++)
            result.at(0, j) += m.at(i, j);
    return result;
}