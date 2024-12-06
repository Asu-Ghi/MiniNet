#ifndef GLOBAL_H
#define GLOBAL_H

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <float.h>

/*
Activation function enum structure
Enum to store what activation function is being used
*/
typedef enum {
    RELU,
    LEAKY_RELU,
    SOFTMAX,
    SIGMOID,
    LINEAR,
    TANH
} ActivationType;

/*
Optimization function enum structure
Enum to store what optimization function to use.
*/
typedef enum {
    SGD,
    SGD_MOMENTUM,
    ADA_GRAD,
    RMS_PROP,
    ADAM
}OptimizationType;

/*
Loss Type Enum
Stores type of loss to use in final layer, stored as a layer param however for access in methods
*/
typedef enum {
    CATCROSSENTROPY,
    BINCROSSENTROPY,
    MSE, // Mean Square Error
    MAE // Mean Absolute Error
} LossType;

/*
Loss Struct
Stores type of loss, pred X, and the double loss calculated
*/
typedef struct {
    LossType lossType;
    matrix* X;
    double loss;
} Loss;

#endif 