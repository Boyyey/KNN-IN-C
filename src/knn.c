#include "knn.h"
#include "simd_math.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

KNNClassifier* knn_create(int k, int use_simd) {
    KNNClassifier* classifier = (KNNClassifier*)malloc(sizeof(KNNClassifier));
    classifier->k = k;
    classifier->use_simd = use_simd;
    classifier->data = NULL;
    classifier->labels = NULL;
    return classifier;
}

void knn_free(KNNClassifier* classifier) {
    if (classifier->data) matrix_free(classifier->data);
    if (classifier->labels) matrix_free(classifier->labels);
    free(classifier);
}

void knn_fit(KNNClassifier* classifier, Matrix* data, Matrix* labels) {
    // Store copies of the data
    if (classifier->data) matrix_free(classifier->data);
    if (classifier->labels) matrix_free(classifier->labels);
    
    classifier->data = matrix_copy(data);
    classifier->labels = matrix_copy(labels);
}

int knn_predict_single(const KNNClassifier* classifier, const float* sample) {
    size_t num_samples = classifier->data->rows;
    size_t num_features = classifier->data->cols;
    
    // Calculate distances to all training samples
    float* distances = (float*)malloc(num_samples * sizeof(float));
    
    if (classifier->use_simd && simd_avx2_supported()) {
        // Create a temporary matrix for the query sample
        Matrix query_sample = {
            .rows = 1,
            .cols = num_features,
            .stride = num_features,
            .data = (float*)sample,
            .is_view = 1
        };
        
        simd_batch_euclidean_distances_avx2(&query_sample, classifier->data, distances);
    } else {
        // Use optimized non-SIMD version
        Matrix query_sample = {
            .rows = 1,
            .cols = num_features,
            .stride = num_features,
            .data = (float*)sample,
            .is_view = 1
        };
        
        optimized_batch_euclidean_distances(&query_sample, classifier->data, distances);
    }
    
    // Get indices of k nearest neighbors
    int* indices = (int*)malloc(num_samples * sizeof(int));
    argsort(distances, num_samples, indices);
    
    // Get labels of k nearest neighbors
    int* neighbor_labels = (int*)malloc(classifier->k * sizeof(int));
    for (int i = 0; i < classifier->k; i++) {
        int idx = indices[i];
        neighbor_labels[i] = (int)classifier->labels->data[idx];
    }
    
    // Find most common label
    int prediction = most_common_label(neighbor_labels, indices, classifier->k);
    
    // Clean up
    free(distances);
    free(indices);
    free(neighbor_labels);
    
    return prediction;
}

void knn_predict_batch(const KNNClassifier* classifier, const Matrix* samples, int* predictions) {
    size_t num_test_samples = samples->rows;
    
    #pragma omp parallel for
    for (size_t i = 0; i < num_test_samples; i++) {
        const float* sample = &samples->data[i * samples->stride];
        predictions[i] = knn_predict_single(classifier, sample);
    }
}

// Utility function to get indices that would sort an array
void argsort(const float* array, size_t size, int* indices) {
    // Initialize indices
    for (size_t i = 0; i < size; i++) {
        indices[i] = i;
    }
    
    // Simple bubble sort (could be optimized with better algorithm)
    for (size_t i = 0; i < size - 1; i++) {
        for (size_t j = 0; j < size - i - 1; j++) {
            if (array[indices[j]] > array[indices[j + 1]]) {
                int temp = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = temp;
            }
        }
    }
}

// Find the most common label among k neighbors
int most_common_label(const int* labels, const int* indices, int k) {
    int max_count = 0;
    int most_common = -1;
    
    for (int i = 0; i < k; i++) {
        int current_label = labels[i];
        int count = 0;
        
        for (int j = 0; j < k; j++) {
            if (labels[j] == current_label) {
                count++;
            }
        }
        
        if (count > max_count) {
            max_count = count;
            most_common = current_label;
        }
    }
    
    return most_common;
}

void knn_benchmark(const KNNClassifier* classifier, const Matrix* test_data, const Matrix* test_labels, int iterations) {
    printf("Running benchmark with %d iterations...\n", iterations);
    
    double total_time = 0.0;
    int correct = 0;
    
    for (int iter = 0; iter < iterations; iter++) {
        clock_t start = clock();
        
        int* predictions = (int*)malloc(test_data->rows * sizeof(int));
        knn_predict_batch(classifier, test_data, predictions);
        
        clock_t end = clock();
        total_time += ((double)(end - start)) / CLOCKS_PER_SEC;
        
        // Calculate accuracy for first iteration
        if (iter == 0) {
            for (size_t i = 0; i < test_data->rows; i++) {
                if (predictions[i] == (int)test_labels->data[i]) {
                    correct++;
                }
            }
        }
        
        free(predictions);
    }
    
    double avg_time = total_time / iterations;
    double accuracy = (double)correct / test_data->rows * 100.0;
    
    printf("Average prediction time: %.4f seconds\n", avg_time);
    printf("Accuracy: %.2f%% (%d/%zu)\n", accuracy, correct, test_data->rows);
    printf("Samples per second: %.2f\n", test_data->rows / avg_time);
}