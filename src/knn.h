#ifndef KNN_H
#define KNN_H

#include "matrix.h"

typedef struct {
    Matrix* data;
    Matrix* labels;
    int k;
    int use_simd;
} KNNClassifier;

// Classifier functions
KNNClassifier* knn_create(int k, int use_simd);
void knn_free(KNNClassifier* classifier);
void knn_fit(KNNClassifier* classifier, Matrix* data, Matrix* labels);

// Prediction functions
int knn_predict_single(const KNNClassifier* classifier, const float* sample);
void knn_predict_batch(const KNNClassifier* classifier, const Matrix* samples, int* predictions);

// Utility functions
void argsort(const float* array, size_t size, int* indices);
int most_common_label(const int* labels, const int* indices, int k);

// Performance comparison
void knn_benchmark(const KNNClassifier* classifier, const Matrix* test_data, const Matrix* test_labels, int iterations);

#endif // KNN_H