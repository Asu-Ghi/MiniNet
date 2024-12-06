#ifndef SOFTMAX_H
#define SOFTMAX_H
#include "linalg.h"

/*
Activation Parameter Structure
Contains activation parameters for a layer
*/
typedef struct {
    ActivationType SOFTMAX; // Activation type to use
    matrix* inputs;
    matrix* dinputs;
    matrix* outputs; // Post activation outputs 
} SoftMaxParams;

/*
Init SoftMax Params,
*/
SoftMaxParams* init_softmax();

/*
Free memory for softmax struct
*/
void free_softmax(SoftMaxParams* softmax);

/*
SoftMax activation forward pass
*/
void softmax_forwards(SoftMaxParams* softmax, matrix* inputs);

/*
SoftMax activation backward pass
Responsible for Freeing inputs after backward pass
*/
void softmax_backwards(SoftMaxParams* softmax, matrix* Y);


#endif