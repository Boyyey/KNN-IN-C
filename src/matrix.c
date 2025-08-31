#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#endif

Matrix* matrix_create(size_t rows, size_t cols) {
    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
    if (!m) return NULL;
    
    m->rows = rows;
    m->cols = cols;
    m->stride = cols;
    m->is_view = 0;
    
    m->data = (float*)calloc(rows * cols, sizeof(float));
    if (!m->data) {
        free(m);
        return NULL;
    }
    
    return m;
}

Matrix* matrix_create_aligned(size_t rows, size_t cols, size_t alignment) {
    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
    if (!m) return NULL;
    
    m->rows = rows;
    m->cols = cols;
    m->stride = cols;
    m->is_view = 0;
    
    // Allocate aligned memory
    #ifdef _ISOC11_SOURCE
    m->data = (float*)aligned_alloc(alignment, rows * cols * sizeof(float));
    #else
    // Fallback to posix_memalign
    if (posix_memalign((void**)&m->data, alignment, rows * cols * sizeof(float)) != 0) {
        free(m);
        return NULL;
    }
    #endif
    
    if (!m->data) {
        free(m);
        return NULL;
    }
    
    // Initialize to zero
    memset(m->data, 0, rows * cols * sizeof(float));
    
    return m;
}

void matrix_free(Matrix* m) {
    if (!m) return;
    
    if (!m->is_view && m->data) {
        free(m->data);
    }
    
    free(m);
}

void matrix_fill(Matrix* m, float value) {
    if (!m || !m->data) return;
    
    #pragma omp parallel for
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            m->data[i * m->stride + j] = value;
        }
    }
}

void matrix_random_fill(Matrix* m, float min, float max) {
    if (!m || !m->data) return;
    
    #pragma omp parallel for
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            float random_val = (float)rand() / RAND_MAX;
            m->data[i * m->stride + j] = min + random_val * (max - min);
        }
    }
}

void matrix_copy(Matrix* dest, const Matrix* src) {
    if (!dest || !src || !dest->data || !src->data) return;
    if (dest->rows != src->rows || dest->cols != src->cols) return;
    
    #pragma omp parallel for
    for (size_t i = 0; i < src->rows; i++) {
        for (size_t j = 0; j < src->cols; j++) {
            dest->data[i * dest->stride + j] = src->data[i * src->stride + j];
        }
    }
}

float matrix_get(const Matrix* m, size_t row, size_t col) {
    if (!m || !m->data || row >= m->rows || col >= m->cols) {
        return 0.0f;
    }
    return m->data[row * m->stride + col];
}

void matrix_set(Matrix* m, size_t row, size_t col, float value) {
    if (!m || !m->data || row >= m->rows || col >= m->cols) {
        return;
    }
    m->data[row * m->stride + col] = value;
}

void matrix_print(const Matrix* m, const char* name) {
    if (!m || !m->data) {
        printf("%s: NULL matrix\n", name);
        return;
    }
    
    printf("%s (%zux%zu):\n", name, m->rows, m->cols);
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            printf("%8.4f ", m->data[i * m->stride + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Additional utility functions not in the header but useful internally
Matrix* matrix_view(Matrix* src, size_t row_start, size_t col_start, size_t rows, size_t cols) {
    if (!src || !src->data) return NULL;
    if (row_start + rows > src->rows || col_start + cols > src->cols) return NULL;
    
    Matrix* view = (Matrix*)malloc(sizeof(Matrix));
    if (!view) return NULL;
    
    view->rows = rows;
    view->cols = cols;
    view->stride = src->stride;
    view->is_view = 1;
    view->data = src->data + row_start * src->stride + col_start;
    
    return view;
}

void matrix_add(Matrix* a, const Matrix* b) {
    if (!a || !b || !a->data || !b->data) return;
    if (a->rows != b->rows || a->cols != b->cols) return;
    
    #pragma omp parallel for
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < a->cols; j++) {
            a->data[i * a->stride + j] += b->data[i * b->stride + j];
        }
    }
}

void matrix_scale(Matrix* m, float scalar) {
    if (!m || !m->data) return;
    
    #pragma omp parallel for
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            m->data[i * m->stride + j] *= scalar;
        }
    }
}

float matrix_sum(const Matrix* m) {
    if (!m || !m->data) return 0.0f;
    
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            sum += m->data[i * m->stride + j];
        }
    }
    
    return sum;
}