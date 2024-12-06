#include "relu.h"

ReluParams* init_relu() {
    ReluParams* relu = malloc(sizeof(ReluParams));
    relu->dinputs = NULL;
    relu->outputs = NULL;
    relu->inputs = NULL;
    return relu;
}

void free_relu(ReluParams* relu) {
    if (relu->dinputs != NULL) {
        free_matrix(relu->dinputs);
    }
    if (relu->outputs != NULL) {
        free_matrix(relu->outputs);
    }
    if (relu->inputs != NULL) {
        free_matrix(relu->inputs);
    }
}

void relu_forwards(ReluParams* relu, matrix* inputs) {
    // Allocate memory for structure variables dynamically
    if (relu->inputs == NULL) {
        relu->inputs = allocate_matrix(inputs->rows, inputs->cols);
    }
    if (relu->dinputs == NULL) {
        relu->dinputs = allocate_matrix(inputs->rows, inputs->cols);
    }
    if (relu->outputs == NULL) {
        relu->outputs = allocate_matrix(inputs->rows, inputs->cols);
    } 
    // temp
    memcpy(relu->inputs->data, inputs->data, inputs->rows * inputs->cols * sizeof(double));
    // Calculate outputs

    #ifdef ENABLE_PARALLEL
    #pragma omp for schedule(static)
    #endif
    for (int i = 0; i < relu->inputs->rows * relu->inputs->cols; i++) {
        relu->outputs->data[i] = (relu->inputs->data[i] <= 0) ? 0 : relu->inputs->data[i];
    }
}

void relu_backwards(ReluParams* relu, matrix* input_gradients) {
    // Check dimensions
    if (relu->inputs->rows != input_gradients->rows || 
        relu->inputs->cols != input_gradients->cols ) {
        fprintf(stderr, "Error, Dimensionality mismatch in backwards relu.\n");
        exit(1);
    }

    // Allocate memory for structure variable dynamically
    if (relu->dinputs == NULL) {
        relu->dinputs = allocate_matrix(input_gradients->rows, input_gradients->cols);
    }

    #ifdef ENABLE_PARALLEL 
        #pragma omp for schedule(static)
    #endif

    // Iterate through every value in layer post activation output to get relu gradients
    for (int i = 0; i < input_gradients->rows * input_gradients->cols; i++) {
        relu->dinputs->data[i] = 
        (relu->inputs->data[i] > 0) ? input_gradients->data[i] : 0;
    }
}
