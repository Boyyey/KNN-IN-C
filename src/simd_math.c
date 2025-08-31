#include "simd_math.h"
#include <math.h>
#include <stdio.h>

int simd_avx2_supported() {
    // In a real implementation, you'd use cpuid to check
    // For simplicity, we'll assume AVX2 is available
    return 1;
}

float simd_euclidean_distance_avx2(const float* a, const float* b, size_t size) {
    __m256 sum_vec = _mm256_setzero_ps();
    size_t i;

    // Process 8 elements at a time
    for (i = 0; i + 7 < size; i += 8) {
        __m256 vec_a = _mm256_loadu_ps(&a[i]);
        __m256 vec_b = _mm256_loadu_ps(&b[i]);
        
        __m256 diff = _mm256_sub_ps(vec_a, vec_b);
        __m256 squared = _mm256_mul_ps(diff, diff);
        
        sum_vec = _mm256_add_ps(sum_vec, squared);
    }

    // Horizontal sum of the vector
    float sum_array[8];
    _mm256_storeu_ps(sum_array, sum_vec);
    float sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] + 
                sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];

    // Process remaining elements
    for (; i < size; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sqrtf(sum);
}

float naive_euclidean_distance(const float* a, const float* b, size_t size) {
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

void simd_batch_euclidean_distances_avx2(const Matrix* query, const Matrix* data, float* distances) {
    size_t num_samples = data->rows;
    size_t num_features = data->cols;
    
    #pragma omp parallel for
    for (size_t i = 0; i < num_samples; i++) {
        const float* sample = &data->data[i * data->stride];
        distances[i] = simd_euclidean_distance_avx2(query->data, sample, num_features);
    }
}

void optimized_batch_euclidean_distances(const Matrix* query, const Matrix* data, float* distances) {
    size_t num_samples = data->rows;
    size_t num_features = data->cols;
    
    // Cache-friendly implementation: process samples in blocks
    const size_t block_size = 64; // Cache line friendly size
    
    #pragma omp parallel for
    for (size_t i = 0; i < num_samples; i += block_size) {
        size_t end = i + block_size < num_samples ? i + block_size : num_samples;
        
        for (size_t j = i; j < end; j++) {
            const float* sample = &data->data[j * data->stride];
            float sum = 0.0f;
            
            // Manual loop unrolling for better performance
            size_t k = 0;
            for (; k + 3 < num_features; k += 4) {
                float diff1 = query->data[k] - sample[k];
                float diff2 = query->data[k+1] - sample[k+1];
                float diff3 = query->data[k+2] - sample[k+2];
                float diff4 = query->data[k+3] - sample[k+3];
                
                sum += diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4;
            }
            
            // Process remaining elements
            for (; k < num_features; k++) {
                float diff = query->data[k] - sample[k];
                sum += diff * diff;
            }
            
            distances[j] = sqrtf(sum);
        }
    }
}