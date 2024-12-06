#include "linalg.h"
//////////////////////////////////////////////////// HELPER FUNCTIONS //////////////////////////////////////////////////////////////

matrix* allocate_matrix(int rows, int cols) {
    matrix* M = malloc(sizeof(matrix));
    M->rows = rows;
    M->cols = cols;
    M->data = (double*) calloc(rows * cols, sizeof(double));

    if (M->data == NULL) {
        fprintf(stderr, "Memory Allocation failed in allocate matrix.\n");
        printf("Expected Dim size = (%d x %d)\n", rows, cols);
        exit(1);
    }
    
    return M;
}

void free_matrix(matrix* M) {
    free(M->data);
    M->data = NULL;
    if (M->data != NULL) {
        fprintf(stderr, "Error: Freeing memory failed in free matrix.\n");
        exit(1);
    }
    free(M);
    M = NULL;
    if (M != NULL) {
        fprintf(stderr, "Error: Freeing memory failed in free matrix.\n");
        exit(1);
    }   
}

void shallow_cpy_matrix(matrix* src, matrix* dest, int start_row, int num_rows) {
    dest->rows = num_rows;
    dest->cols = src->cols;
    dest->data = src->data + start_row * dest->cols; // Point to the starting row
}

void print_matrix(matrix* M) {
    int m = M->rows;  // Number of rows
    int n = M->cols;  // Number of columns
    // Print dim
    printf("Dim:(%d x %d)\n", m, n);    
    // Loop through the rows and columns of the matrix
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", M->data[i * n + j]);
        }
        printf("\n");  // New line after each row
    }
}

void fill_matrix(matrix* M, double val) {
    if (M->data == NULL) {
        fprintf(stderr, "Matrix Data not allocated in fill matrix.\n");
        exit(1);
    }

    for (int i = 0; i < M->rows; i++) {
        for (int j = 0; j < M->cols; j++) {
            M->data[i * M->cols + j] = val;
        }
    }
}

//////////////////////////////////////////////////// LIN ALG FUNCTIONS //////////////////////////////////////////////////////////////

matrix* transpose_matrix(matrix* w){

    // Check w memory
    if (w->data == NULL) {
        fprintf(stderr, "Error: Input Matrix has no data (NULL).\n");
        exit(1);
    }

    // Create a new matrix object to hold the transposed matrix
    matrix* transposed_matrix = (matrix*) malloc(sizeof(matrix));

    // Check memory allocation for the matrix struct
    if (transposed_matrix == NULL) {
        fprintf(stderr, "Error: Memory allocation failure for transposed_matrix struct.\n");
        exit(1);
    }

    // Allocate memory for the transposed data
    transposed_matrix->rows = w->cols;  // Transposed matrix rows = original matrix cols
    transposed_matrix->cols = w->rows;  // Transposed matrix cols = original matrix rows
    transposed_matrix->data = (double*) calloc(transposed_matrix->rows * transposed_matrix->cols, sizeof(double));

    // Check memory allocation for the transposed data
    if (transposed_matrix->data == NULL) {
        fprintf(stderr, "Error: Memory allocation failure for transposed matrix data.\n");
        exit(1);
    }

    // Iterate through the original matrix and fill the transposed matrix
    for (int i = 0; i < w->rows; i++) {
        for (int j = 0; j < w->cols; j++) {
            // Swap row and column indices to transpose the matrix
            transposed_matrix->data[j * w->rows + i] = w->data[i * w->cols + j];
        }
    }

    // Return the pointer to the transposed matrix
    return transposed_matrix;
}

matrix* matrix_mult(matrix* w, matrix* v) {

    // Get dimensionality info
    int rows_w = w->rows;
    int cols_w = w->cols;
    int cols_v = v->cols;

    // Check dimensions
    if (w->cols != v->rows) {
        fprintf(stderr, "Error in matrix mult, dimensionality mismatch.\n");
        exit(1);
    }

    // Allocate result matrix with dimensions rows_w x cols_v
    matrix* result = malloc(sizeof(matrix));
    result->rows = rows_w;
    result->cols = cols_v;
    result->data = (double*) calloc(rows_w * cols_v, sizeof(double));
    
    // Check memory allocation
    if (result->data == NULL) {
        fprintf(stderr, "Error: Memory allocation failure in matrix_mult.\n");
        exit(1);
    }

#ifdef ENABLE_PARALLEL
    int block_size = 32;
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < rows_w; i += block_size) {
        for (int j = 0; j < cols_v; j += block_size) {
            // Tile multiplication
            for (int k = 0; k < cols_w; k += block_size) {
                // Iterate over blocks of w, v, and result
                for (int ii = i; ii < i + block_size && ii < rows_w; ++ii) {
                    for (int jj = j; jj < j + block_size && jj < cols_v; ++jj) {
                        double sum = 0.0;
                        for (int kk = k; kk < k + block_size && kk < cols_w; ++kk) {
                            sum += w->data[ii * cols_w + kk] * v->data[kk * cols_v + jj];
                        }
                        result->data[ii * cols_v + jj] += sum;
                    }
                }
            }
        }
    }
#else 
        for (int i = 0; i < rows_w; i++) {
            for (int j = 0; j < cols_v; j++) {
                for (int k = 0; k < cols_w; k++) {
                    result->data[i * cols_v + j] += w->data[i * cols_w + k] * v->data[k * cols_v + j];
                }
            }
        }


#endif

    return result;

}

matrix* element_matrix_mult(matrix* w, matrix* v){
    // Check dimensions
    if(w->rows != v->rows || w->cols != v->cols) {
        fprintf(stderr, "Error, mismatching dimensions in element matrix mult.\n");
    }
    int row_w = w->rows;
    int col_w = w->cols;

    // Allocate and check memory for result
    matrix* result = malloc(sizeof(matrix));
    result->rows = row_w;
    result->cols = col_w;
    result->data = (double*) calloc(row_w * col_w, sizeof(double));

    if (result->data == NULL) {
        fprintf(stderr, "Memory allocation failure for result in element matrix mult.\n");
        exit(1);
    }



#ifdef ENABLE_PARALLEL 
    // Parallel Code
    #pragma omp parallel 
    {
        int thread_id = omp_get_thread_num(); // Get current thread id
        int total_threads = omp_get_num_threads(); // Get total num threads
        int rows_per_thread = (row_w + total_threads - 1) / total_threads; // Get num rows to calc per each thread
        int start_row = rows_per_thread * thread_id; // Get start row for unique thread
        int end_row = rows_per_thread * thread_id + rows_per_thread; // Get end row for unique thread

        if (end_row > row_w) {
            end_row = row_w;
        }

        for (int i = start_row; i < end_row; i++) {
            for (int j = 0; j < col_w; j++) {
                result->data[i * col_w + j] = w->data[i * col_w + j] * v->data[i * col_w + j];
            }
        }
    }

#else

    // Sequential Code
    for (int i = 0; i < row_w; i++) {
        for (int j = 0; j < col_w; j++) {
            result->data[i * col_w + j] = w->data[i * col_w + j] * v->data[i * col_w + j];
        }
    }

#endif
    return result;
}

void matrix_scalar_mult(matrix* w, double s) {

    int rows = w->rows;
    int cols = w->cols;
    
#ifdef ENABLE_PARALLEL
#pragma omp parallel
{
    int thread_id = omp_get_thread_num(); // Get current thread id
    int total_threads = omp_get_num_threads(); // Get total num threads
    int rows_per_thread = (rows + total_threads - 1) / total_threads; // Get num rows to calc per each thread
    int start_row = rows_per_thread * thread_id; // Get start row for unique thread
    int end_row = rows_per_thread * thread_id + rows_per_thread; // Get end row for unique thread
    
    // check bounds
    if(end_row > rows) {
        end_row = rows;
    }

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < cols; j++) {
            w->data[i * cols + j] = s * w->data[i * cols + j];
        }
    }

}

#else
    for (int i = 0; i < rows * cols; i++) {
        w->data[i] = s * w->data[i];
    }

#endif

}

matrix* matrix_sum(matrix* w, matrix* v) {

    // Check dimensions
    if (w->rows != v->rows || w->cols != v->cols) {
        fprintf(stderr, "Error, Dimensionality Mismatch in Matrix Sum.\n");
        exit(1);
    }

    // Allocate memory for the return object
    matrix* result = malloc(sizeof(matrix));
    result->rows = w->rows;
    result->cols = w->cols;
    result->data = (double*) calloc(result->rows * result->cols, sizeof(double));
    
    // Check memory allocation
    if (result->data == NULL) {
        fprintf(stderr, "Error: Memory allocation failed in matrix sum.\n");
        exit(1);
    }

    // Get dimensions
    int row_w = w->rows;
    int col_w = w->cols;

// Parallel Code
#ifdef ENABLE_PARALLEL
    #pragma omp parallel
    {   
        // Init parallel constraints
        int thread_id = omp_get_thread_num(); // Get current thread id
        int total_threads = omp_get_num_threads(); // Get total num threads
        int rows_per_thread = (row_w + total_threads - 1) / total_threads; // Get num rows to calc per each thread
        int start_row = rows_per_thread * thread_id; // Get start row for unique thread
        int end_row = rows_per_thread * thread_id + rows_per_thread; // Get end row for unique thread

        // Check bounds
        if (end_row > row_w) {
            end_row = row_w;
        }

        // Parallel, each thread gets range from start to end row.
        for (int i = start_row; i < end_row; i++) {
            for (int j = 0; j < col_w; j++) {
                result->data[i * col_w + j] = w->data[i * col_w + j] + v->data[i * col_w + j]; // Row major order
            }
        }
    }


// Sequential Code
#else

    for (int i = 0; i < row_w; i++) {
        for (int j = 0; j < col_w; j++) {
            result->data[i * col_w + j] = w->data[i * col_w + j] + v->data[i * col_w + j]; // Row major order
        }
    }

#endif

    return result;
}

matrix* matrix_scalar_sum(matrix* w, double s, bool useAbs) {

    // Allocate memory for the result
    matrix* result = malloc(sizeof(matrix));
    result->rows = w->rows;
    result->cols = w->cols;

    // Check memory
    if (result->data == NULL) {
        fprintf(stderr, "Error: Memory allocation failed in matrix scalar sum\n");
        exit(1);
    }


#ifdef ENABLE_PARALLEL // Parallel approach

    #pragma omp for schedule(static) // No race conditions, each thread gets its own i
    for (int i = 0; i < result->rows * result->cols; i++) {
        if (useAbs){
            result->data[i] = fabs(w->data[i] + s); // useAbs allows for more control.
        }
        else {
            result->data[i] = w->data[i] + s;
        }
    }


#else // Sequential Approach
    for (int i = 0; i < result->rows * result->cols; i++) {
        if (useAbs){
            result->data[i] = fabs(w->data[i] + s);
        }
        else {
            result->data[i] = w->data[i] + s;
        }
    }

#endif

    return result; // return pointer to matrix
}

double matrix_mean(matrix* w) {
    int rows = w->rows;
    int cols = w->cols;
    double sum = 0.0;

#ifdef ENABLE_PARALLEL
    #pragma omp for collapse(2) schedule(static)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sum += w->data[i * cols + j];
        }
    }
    sum = sum / (rows * cols);

#else
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sum += w->data[i * cols + j];
        }
    }
    sum = sum / (rows * cols);

#endif
    return sum;
}