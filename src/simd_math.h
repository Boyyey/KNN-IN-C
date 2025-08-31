#ifndef SIMD_MATH_H
#define SIMD_MATH_H

#include <immintrin.h> // AVX2 intrinsics
#include "matrix.h"

// Check if AVX2 is supported
int simd_avx2_supported();

// Euclidean distance with AVX2
float simd_euclidean_distance_avx2(const float* a, const float* b, size_t size);

// Euclidean distance without SIMD (for comparison)
float naive_euclidean_distance(const float* a, const float* b, size_t size);

// Batch Euclidean distances with AVX2
void simd_batch_euclidean_distances_avx2(const Matrix* query, const Matrix* data, float* distances);

// Batch Euclidean distances with cache optimization
void optimized_batch_euclidean_distances(const Matrix* query, const Matrix* data, float* distances);

#endif // SIMD_MATH_H