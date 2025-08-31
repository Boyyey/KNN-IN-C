#include "../src/knn.h"
#include "../src/matrix.h"
#include <stdio.h>
#include <time.h>

// Simple Iris dataset loader (in real implementation, load from CSV)
void load_iris_data(Matrix** data, Matrix** labels, float train_ratio) {
    // This would normally load from a file
    // For demonstration, we'll create dummy data
    int num_samples = 150;
    int num_features = 4;
    int num_classes = 3;
    
    *data = matrix_create(num_samples, num_features);
    *labels = matrix_create(num_samples, 1);
    
    // Fill with random data resembling Iris dataset
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_features; j++) {
            float value;
            if (j == 0) value = 4.0 + (float)rand() / RAND_MAX * 4.0; // sepal length
            else if (j == 1) value = 2.0 + (float)rand() / RAND_MAX * 3.0; // sepal width
            else if (j == 2) value = 1.0 + (float)rand() / RAND_MAX * 6.0; // petal length
            else value = 0.1 + (float)rand() / RAND_MAX * 2.5; // petal width
            
            matrix_set(*data, i, j, value);
        }
        matrix_set(*labels, i, 0, rand() % num_classes);
    }
}

int main() {
    srand(time(NULL));
    
    printf("=== Blazingly Fast k-NN Implementation ===\n");
    
    // Load data
    Matrix* train_data, * train_labels;
    Matrix* test_data, * test_labels;
    
    load_iris_data(&train_data, &train_labels, 0.8);
    load_iris_data(&test_data, &test_labels, 0.2);
    
    printf("Training samples: %zu\n", train_data->rows);
    printf("Test samples: %zu\n", test_data->rows);
    printf("Features: %zu\n", train_data->cols);
    
    // Create and train k-NN classifier with SIMD
    KNNClassifier* classifier = knn_create(3, 1); // k=3, use SIMD
    knn_fit(classifier, train_data, train_labels);
    
    printf("\n--- SIMD Accelerated k-NN ---\n");
    knn_benchmark(classifier, test_data, test_labels, 5);
    
    // Test without SIMD for comparison
    KNNClassifier* classifier_no_simd = knn_create(3, 0); // k=3, no SIMD
    knn_fit(classifier_no_simd, train_data, train_labels);
    
    printf("\n--- Standard k-NN (No SIMD) ---\n");
    knn_benchmark(classifier_no_simd, test_data, test_labels, 5);
    
    // Clean up
    knn_free(classifier);
    knn_free(classifier_no_simd);
    matrix_free(train_data);
    matrix_free(train_labels);
    matrix_free(test_data);
    matrix_free(test_labels);
    
    return 0;
}