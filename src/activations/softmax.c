#include "softmax.h"

SoftMaxParams* init_softmax() {
    SoftMaxParams* softmax = malloc(sizeof(SoftMaxParams));
    softmax->inputs = NULL;
    softmax->dinputs = NULL;
    softmax->outputs = NULL;
    return softmax;
}

void free_softmax(SoftMaxParams* softmax) {
    if (softmax->inputs != NULL) {
        free_matrix(softmax->inputs);
    }
    if (softmax->dinputs != NULL) {
        free_matrix(softmax->dinputs);
    }   
    if (softmax->outputs != NULL) {
        free_matrix(softmax->outputs);
    }
}

void softmax_forwards(SoftMaxParams* softmax, matrix* inputs) {
    // Allocate memory for structure variables dynamically
    if (softmax->inputs == NULL) {
        softmax->inputs = allocate_matrix(inputs->rows, inputs->cols);
    }
    if (softmax->dinputs == NULL) {
        softmax->dinputs = allocate_matrix(inputs->rows, inputs->cols);
    }
    if (softmax->outputs == NULL) {
        softmax->outputs = allocate_matrix(inputs->rows, inputs->cols);
    }
    // temp
    memcpy(softmax->inputs->data, inputs->data, inputs->rows * inputs->cols * sizeof(double));

    // Calculate softmax for every sample in batch
    for(int i = 0; i < inputs->rows; i++) {

        // Subtract maximum value from each value in the input batch for numerical stability
        double max = -DBL_MAX;
        for(int j = 0; j < inputs->cols; j++) {
            if (inputs->data[i * inputs->cols + j] > max) {
                max = inputs->data[i * inputs->cols + j];
            }
        }

        // Calculate exponentials and sum them
        double* exp_values = (double*) calloc(inputs->cols, sizeof(double));
        double sum = 0.0;
        #ifdef ENABLE_PARALLEL
        #pragma omp parallel for reduction(+:sum)
        #endif
        for(int j = 0; j < inputs->cols; j++) {
            exp_values[j] = exp(inputs->data[i * inputs->cols + j] - max);
            sum += exp_values[j];
        }

        // Normalize exponentials by dividing by the sum to get probabilities
        #ifdef ENABLE_PARALLEL
        #pragma omp parallel for
        #endif
        for(int j = 0; j < inputs->cols; j++) {
            softmax->outputs->data[i * inputs->cols + j] = exp_values[j] / sum;
        }

        // free exp memory
        free(exp_values);
    }
}

void softmax_backwards(SoftMaxParams* softmax, matrix* Y) {
    // Check dimensions
    if (softmax->outputs->rows != Y->rows || softmax->outputs->cols != Y->cols) {
        fprintf(stderr, "Error: Dimensionality mismatch in softmax backwards.\n");
        exit(1);
    }

    #ifdef ENABLE_PARALLEL
    #pragma omp parallel for
    #endif
    for (int i = 0; i < softmax->dinputs->rows; i++) {
        int row_offset = i * softmax->dinputs->cols;  
        for (int j = 0; j < softmax->dinputs->cols; j++) {
            softmax->dinputs->data[row_offset + j] = softmax->outputs->data[row_offset + j] - Y->data[row_offset + j];
        }
    }
}


