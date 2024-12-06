#ifndef RELU_H
#define RELU_H
#include "linalg.h"

/*
Activation Parameter Structure
Contains activation parameters for a layer
*/
typedef struct {
    matrix* inputs;
    matrix* dinputs;
    matrix* outputs; // Post activation outputs 
} ReluParams;

/*
Init ReLu Params,
*/
ReluParams* init_relu();

/*
Free ReLu struct
*/
void free_relu(ReluParams* relu);

/*
ReLU activation forward pass
*/
void relu_forwards(ReluParams* relu, matrix* inputs);

/*
ReLU activation backward pass
Responsible for Freeing inputs after backward pass
*/
void relu_backwards(ReluParams* relu, matrix* input_gradients);


#endif