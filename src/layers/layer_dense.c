#include "layer_dense.h"

layer_dense* init_layer(int num_inputs, int num_neurons) {
    layer_dense* layer = malloc(sizeof(layer_dense));
    layer->num_inputs = num_inputs;
    layer->num_neurons = num_neurons;

    layer->inputs = NULL; // default
    layer->dinputs = NULL; // default
    layer->outputs = NULL; // default

    layer->weights = allocate_matrix(num_inputs, num_neurons);
    layer->dweights = allocate_matrix(num_inputs, num_neurons);

    layer->biases = allocate_matrix(1, num_neurons);
    layer->dbiases = allocate_matrix(1, num_neurons);

    layer->useRegularization = false; // default
    layer->lambda_l1 = 5e-4; // default
    layer->lambda_l2 = 5e-4; // default
    layer->id = -1; // default

    // Initialize Weights
    // srand(time(NULL));  // Seed random number with current time
    srand(42);
    for (int i = 0; i < num_neurons * num_inputs; i++){

        // He initialization
        layer->weights->data[i] = sqrt(1.0 / num_inputs) * ((double)rand() / RAND_MAX * 2.0 - 1.0);  

        // Xavier init
        // layer_->weights->data[i] = sqrt(1.0 / (num_inputs + num_neurons)) * ((double)rand() / RAND_MAX * 2.0 - 1.0);
    }
    return layer;
}

void free_layer(layer_dense* layer) {
    // Free weights
    free(layer->weights->data);
    free(layer->weights);
    layer->weights = NULL;

    // Free biases
    free(layer->biases->data);
    free(layer->biases);
    layer->biases = NULL;

    // Free dweights
    if (layer->dweights != NULL) {
        free(layer->dweights->data);
        free(layer->dweights); 
        layer->dweights = NULL; 
    }

    // Free dbiases
    if (layer->dbiases != NULL) {
        free(layer->dbiases->data);
        free(layer->dbiases);     
        layer->dbiases = NULL;
    }

}

void clean_memory_forward(layer_dense* layer) {

    if (layer->inputs != NULL) {
        free_matrix(layer->inputs);
    }
    if (layer->dinputs != NULL) {
        free_matrix(layer->dinputs);
    }
    if (layer->dweights != NULL ) {
        free_matrix(layer->dweights);
    }
    if (layer->dbiases != NULL ) {
        free_matrix(layer->dbiases);
    }
}

void dense_forwards(matrix* inputs, layer_dense* layer) {
    // Allocate memory for layer input 
    if (layer->inputs == NULL) {
        layer->inputs = allocate_matrix(inputs->rows, inputs->cols);
    } 

    // Allocate memory for derivative of inputs
    if (layer->dinputs == NULL) {
        layer->dinputs = allocate_matrix(inputs->rows, inputs->cols);
    }

    // Copy inputs into layer structure
    memcpy(layer->inputs->data, inputs->data, layer->inputs->rows * layer->inputs->cols * sizeof(double));

    // Allocate memory for pre activation outputs
    if (layer->outputs == NULL) {
        layer->outputs = allocate_matrix(inputs->rows, layer->num_neurons);
    }
    
    // Calculate Z
    matrix* mult_matrix = matrix_mult(inputs, layer->weights); // supports parallel

    // Add biases for the layer to the batch output data
    #pragma omp for collapse(2) schedule(static)
    for (int i = 0; i < layer->outputs->rows; i++) {
        // output dim2-> num neurons
        for (int j = 0; j < layer->outputs->cols; j++) {
            layer->outputs->data[i * layer->outputs->cols + j] = 
            mult_matrix->data[i * layer->outputs->cols + j] + layer->biases->data[j];
        }
    }

    // Free uneeded memory
    free_matrix(mult_matrix);}

void dense_backwards(matrix* input_gradients, layer_dense* layer) {
    
    // Calculate weight gradients
    matrix* inputs_transposed = transpose_matrix(layer->inputs);

    // Check dimensions
    if(inputs_transposed->cols != input_gradients-> rows) {
        fprintf(stderr, "Error: Dimensionality mismatch (inputs transposed backward dense).\n");
        exit(1);
    }
    layer->dweights = matrix_mult(inputs_transposed, input_gradients); 

    // Calculate bias gradients
    for (int j = 0; j < layer->dbiases->cols; j++) {
        for(int i = 0; i < input_gradients->rows; i++) {
            // sum across rows
            layer->dbiases->data[j] += input_gradients->data[i * input_gradients->cols + j];
        }
    }

    // Calculate regularization gradients if using
    if (layer->useRegularization) {
        calculate_reg_gradients(layer);
    }

    // Calculate input gradients
    matrix* weights_transposed = transpose_matrix(layer->weights);
 
     // Check dimensions
    if (input_gradients->cols != weights_transposed->rows) {
        fprintf(stderr, "Error: Dimensionality mismatch (weights transposed) in backwards dense.\n");
        exit(1);
    }
    layer->dinputs = matrix_mult(input_gradients, weights_transposed); // supports parallel
    
    free_matrix(inputs_transposed);
    free_matrix(weights_transposed);
}

void calculate_reg_gradients(layer_dense* layer) {
#ifdef ENABLE_PARALLEL

    // weights
    #pragma omp for schedule(static)
    for (int i = 0; i < layer->dweights->rows * layer->dweights->cols; i++) {
        // L2 gradients
        layer->dweights->data[i] += 2 * layer->lambda_l2 * layer->weights->data[i];

        // L1 gradients (1 if > 0, -1 if < 0)
        layer->dweights->data[i] += layer->lambda_l1 * (layer->weights->data[i] >= 0.0 ? 1.0 : -1.0);
    }
    // biases
    #pragma omp for schedule(static)
    for (int i = 0; i < layer->dbiases->rows * layer->dbiases->cols; i++) {
        // L2 gradients
        layer->dbiases->data[i] += 2 * layer->lambda_l2 * (layer->biases->data[i]);

        // L1 gradients (1 if > 0, -1 if < 0)
        layer->dbiases->data[i] += layer->lambda_l1 * (layer->biases->data[i] >= 0 ? 1.0: -1.0);
    }

#else
    // weights
    for (int i = 0; i < layer->dweights->rows * layer->dweights->cols; i++) {
        // L2 gradients
        layer->dweights->data[i] += 2 * layer->lambda_l2 * layer->weights->data[i];

        // L1 gradients (1 if > 0, -1 if < 0)
        layer->dweights->data[i] += layer->lambda_l1 * (layer->weights->data[i] >= 0.0 ? 1.0 : -1.0);
    }
    // biases
    for (int i = 0; i < layer->dbiases->rows * layer->dbiases->cols; i++) {
        // L2 gradients
        layer->dbiases->data[i] += 2 * layer->lambda_l2 * (layer->biases->data[i]);

        // L1 gradients (1 if > 0, -1 if < 0)
        layer->dbiases->data[i] += layer->lambda_l1 * (layer->biases->data[i] >= 0 ? 1.0: -1.0);
    }

#endif
}

void calculate_bias_gradients(layer_dense* layer, matrix* input_gradients) {

    // Check dimensions
    if (layer->dbiases->cols != input_gradients->cols) {
        fprintf(stderr, "Dimensionality mismatch in calculate bias gradients.\n");
        exit(1);
    }

#ifdef ENABLE_PARALLEL
    int num_biases = layer->dbiases->cols;
    int row_gradients = input_gradients->rows;
    int col_gradients = input_gradients->cols;

#pragma omp parallel 
{
    int thread_id = omp_get_thread_num(); // Get current thread id
    int total_threads = omp_get_num_threads(); // Get total num threads
    int rows_per_thread = (num_biases + total_threads - 1) / total_threads; // Get num rows to calc per each thread
    int start_row = rows_per_thread * thread_id; // Get start row for unique thread
    int end_row = rows_per_thread * thread_id + rows_per_thread; // Get end row for unique thread

    if (end_row > num_biases) {
        end_row = num_biases;
    }

    // Calculate bias gradients
    for (int j = start_row; j < end_row; j++) {
        for(int i = 0; i < row_gradients; i++) {
            // sum across rows
            layer->dbiases->data[j] += input_gradients->data[i * col_gradients + j];
        }
    }
}
#else

    // Calculate bias gradients
    for (int j = 0; j < layer->dbiases->cols; j++) {
        for(int i = 0; i < input_gradients->rows; i++) {
            // sum across rows
            layer->dbiases->data[j] += input_gradients->data[i * input_gradients->cols + j];
        }
    }

#endif
}
