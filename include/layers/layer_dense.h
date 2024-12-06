#ifndef LAYER_DENSE_H
#define LAYER_DENSE_H
#include "linalg.h"
#include "global.h"
//////////////////////////////////////////////////// DATA STRUCTURES ///////////////////////////////////////////////////////////////////////////

/*
Layer Dense data structure. 
*/
typedef struct {
    int id; // Integer id of layer
    bool is_training; // flag to differentiate between validate and train pass

    int num_neurons; // Number of Neurons in a layer
    int num_inputs; // Number of Input Features into a layer

    matrix* weights; // Layer Weights
    matrix* biases; // Layer Biases
    matrix* inputs; // Inputs used for training

    matrix* dweights; // Gradients for weights
    matrix* dbiases; // Gradients for biases
    matrix* dinputs; // Gradients for inputs

    matrix* outputs; // Outputs used for training (before activation)

    bool useRegularization; // Determines if using L1 and L2 regularization
    double lambda_l1;  // L1 regularization coefficient
    double lambda_l2;  // L2 regularization coefficient 

}layer_dense;

//////////////////////////////////////////////////// LAYER METHODS ///////////////////////////////////////////////////////////////////////////

/*
Initialize a Layer Object
*/
layer_dense* init_layer(int num_inputs, int num_neurons);

/*
Frees all layer dense memory.
*/
void free_layer(layer_dense* layer);

/*
Cleans uneeded memory after training pass
*/
void clean_memory_forward(layer_dense* layer);

/*
Forward Pass for a Dense Layer
*/
void dense_forwards(matrix* inputs, layer_dense* layer);

/*
Backward pass for dense layer
*/
void dense_backwards(matrix* input_gradients, layer_dense* layer);

/*
Calculate Gradients for Backward Pass for Regularization
*/
void calculate_reg_gradients(layer_dense* layer);


/*
Calculate Gradients for Backward Pass for Biases
*/
void calculate_bias_gradients(layer_dense* layer, matrix* input_gradients);

#endif