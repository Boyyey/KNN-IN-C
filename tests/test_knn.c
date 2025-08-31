#include "../src/knn.h"
#include "../src/matrix.h"
#include "../src/simd_math.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

void test_matrix_creation() {
    printf("Testing matrix creation...\n");
    
    Matrix* m = matrix_create(3, 4);
    assert(m != NULL);
    assert(m->rows == 3);
    assert(m->cols == 4);
    assert(m->stride == 4);
    assert(m->is_view == 0);
    assert(m->data != NULL);
    
    // Test that matrix is initialized to zeros
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 4; j++) {
            assert(fabs(matrix_get(m, i, j)) < 1e-6);
        }
    }
    
    matrix_free(m);
    printf("Matrix creation test passed!\n");
}

void test_matrix_operations() {
    printf("Testing matrix operations...\n");
    
    Matrix* a = matrix_create(2, 3);
    Matrix* b = matrix_create(2, 3);
    
    // Fill matrices with test values
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float b_data[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    
    memcpy(a->data, a_data, sizeof(a_data));
    memcpy(b->data, b_data, sizeof(b_data));
    
    // Test matrix addition
    matrix_add(a, b);
    
    assert(fabs(matrix_get(a, 0, 0) - 8.0f) < 1e-6);
    assert(fabs(matrix_get(a, 0, 1) - 10.0f) < 1e-6);
    assert(fabs(matrix_get(a, 0, 2) - 12.0f) < 1e-6);
    assert(fabs(matrix_get(a, 1, 0) - 14.0f) < 1e-6);
    assert(fabs(matrix_get(a, 1, 1) - 16.0f) < 1e-6);
    assert(fabs(matrix_get(a, 1, 2) - 18.0f) < 1e-6);
    
    // Test matrix scaling
    matrix_scale(a, 0.5f);
    
    assert(fabs(matrix_get(a, 0, 0) - 4.0f) < 1e-6);
    assert(fabs(matrix_get(a, 0, 1) - 5.0f) < 1e-6);
    assert(fabs(matrix_get(a, 0, 2) - 6.0f) < 1e-6);
    assert(fabs(matrix_get(a, 1, 0) - 7.0f) < 1e-6);
    assert(fabs(matrix_get(a, 1, 1) - 8.0f) < 1e-6);
    assert(fabs(matrix_get(a, 1, 2) - 9.0f) < 1e-6);
    
    matrix_free(a);
    matrix_free(b);
    printf("Matrix operations test passed!\n");
}

void test_euclidean_distance() {
    printf("Testing Euclidean distance calculations...\n");
    
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    
    // Test naive implementation
    float naive_dist = naive_euclidean_distance(a, b, 3);
    float expected = sqrtf((1-4)*(1-4) + (2-5)*(2-5) + (3-6)*(3-6));
    assert(fabs(naive_dist - expected) < 1e-6);
    
    // Test SIMD implementation if available
    if (simd_avx2_supported()) {
        float simd_dist = simd_euclidean_distance_avx2(a, b, 3);
        assert(fabs(simd_dist - expected) < 1e-6);
        assert(fabs(simd_dist - naive_dist) < 1e-6);
    }
    
    printf("Euclidean distance test passed!\n");
}

void test_knn_fit_predict() {
    printf("Testing k-NN fit and predict...\n");
    
    // Create simple training data
    Matrix* train_data = matrix_create(4, 2);
    Matrix* train_labels = matrix_create(4, 1);
    
    // Two classes: class 0 at (0,0) and (0,1), class 1 at (1,0) and (1,1)
    matrix_set(train_data, 0, 0, 0.0f); matrix_set(train_data, 0, 1, 0.0f);
    matrix_set(train_data, 1, 0, 0.0f); matrix_set(train_data, 1, 1, 1.0f);
    matrix_set(train_data, 2, 0, 1.0f); matrix_set(train_data, 2, 1, 0.0f);
    matrix_set(train_data, 3, 0, 1.0f); matrix_set(train_data, 3, 1, 1.0f);
    
    matrix_set(train_labels, 0, 0, 0.0f);
    matrix_set(train_labels, 1, 0, 0.0f);
    matrix_set(train_labels, 2, 0, 1.0f);
    matrix_set(train_labels, 3, 0, 1.0f);
    
    // Create and train classifier
    KNNClassifier* classifier = knn_create(3, 0); // k=3, no SIMD for deterministic testing
    knn_fit(classifier, train_data, train_labels);
    
    // Test points that should be clearly in one class or another
    float test_point1[] = {0.1f, 0.1f}; // Should be class 0
    float test_point2[] = {0.9f, 0.9f}; // Should be class 1
    float test_point3[] = {0.4f, 0.6f}; // Borderline case
    
    int pred1 = knn_predict_single(classifier, test_point1);
    int pred2 = knn_predict_single(classifier, test_point2);
    int pred3 = knn_predict_single(classifier, test_point3);
    
    assert(pred1 == 0);
    assert(pred2 == 1);
    // pred3 could be either 0 or 1 depending on distance calculation
    
    // Test batch prediction
    Matrix* test_data = matrix_create(2, 2);
    matrix_set(test_data, 0, 0, 0.1f); matrix_set(test_data, 0, 1, 0.1f);
    matrix_set(test_data, 1, 0, 0.9f); matrix_set(test_data, 1, 1, 0.9f);
    
    int* predictions = (int*)malloc(2 * sizeof(int));
    knn_predict_batch(classifier, test_data, predictions);
    
    assert(predictions[0] == 0);
    assert(predictions[1] == 1);
    
    // Clean up
    free(predictions);
    matrix_free(test_data);
    knn_free(classifier);
    matrix_free(train_data);
    matrix_free(train_labels);
    
    printf("k-NN fit and predict test passed!\n");
}

void test_argsort() {
    printf("Testing argsort function...\n");
    
    float values[] = {3.0f, 1.0f, 4.0f, 2.0f};
    int indices[4];
    
    argsort(values, 4, indices);
    
    // Should be sorted as: 1.0, 2.0, 3.0, 4.0
    // So indices should be: 1, 3, 0, 2
    assert(indices[0] == 1);
    assert(indices[1] == 3);
    assert(indices[2] == 0);
    assert(indices[3] == 2);
    
    printf("Argsort test passed!\n");
}

void test_most_common_label() {
    printf("Testing most common label function...\n");
    
    int labels[] = {0, 1, 0, 1, 1};
    int indices[] = {0, 1, 2, 3, 4}; // All indices
    
    // With k=3, should get the most common among first 3: {0, 1, 0} -> 0
    int result1 = most_common_label(labels, indices, 3);
    assert(result1 == 0);
    
    // With k=5, should get the most common among all: {0, 1, 0, 1, 1} -> 1
    int result2 = most_common_label(labels, indices, 5);
    assert(result2 == 1);
    
    printf("Most common label test passed!\n");
}

void test_simd_availability() {
    printf("Testing SIMD availability...\n");
    
    int supported = simd_avx2_supported();
    printf("AVX2 supported: %s\n", supported ? "yes" : "no");
    
    // This test just checks that the function doesn't crash
    // Actual SIMD support depends on the CPU
    printf("SIMD availability test passed!\n");
}

void test_matrix_aligned() {
    printf("Testing aligned matrix creation...\n");
    
    Matrix* m = matrix_create_aligned(10, 10, 32); // 32-byte alignment for AVX
    assert(m != NULL);
    assert(m->data != NULL);
    
    // Check if the pointer is aligned (address divisible by 32)
    uintptr_t address = (uintptr_t)m->data;
    assert(address % 32 == 0);
    
    matrix_free(m);
    printf("Aligned matrix test passed!\n");
}

int main() {
    printf("Running k-NN and matrix tests...\n");
    
    test_matrix_creation();
    test_matrix_operations();
    test_matrix_aligned();
    test_euclidean_distance();
    test_argsort();
    test_most_common_label();
    test_knn_fit_predict();
    test_simd_availability();
    
    printf("All tests passed!\n");
    return 0;
}