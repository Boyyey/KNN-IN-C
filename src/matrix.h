#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>
#include <stdalign.h>

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float *data;
    int is_view;
} Matrix;

// Matrix creation and destruction
Matrix* matrix_create(size_t rows, size_t cols);
Matrix* matrix_create_aligned(size_t rows, size_t cols, size_t alignment);
void matrix_free(Matrix* m);

// Data management
void matrix_fill(Matrix* m, float value);
void matrix_random_fill(Matrix* m, float min, float max);
void matrix_copy(Matrix* dest, const Matrix* src);

// Utility functions
float matrix_get(const Matrix* m, size_t row, size_t col);
void matrix_set(Matrix* m, size_t row, size_t col, float value);
void matrix_print(const Matrix* m, const char* name);

#endif // MATRIX_H