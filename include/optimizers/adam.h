#ifndef ADAM_H
#define ADAM_H
#include "linalg.h"
#include "layer_dense.h"
#include "layer_cnn.h"

/*
Optimization Parameter Structure
Contains optimization parameters for a layer
*/

typedef struct {
    matrix* w_momentums; // Momentums for Weight
    matrix* w_cache; // Cache for Weight
    matrix* b_momentums; // Momentums for Bias
    matrix* b_cache; // Cache for Bias
    double beta_1; // Beta 1 HyperParam
    double beta_2; // Beta 2 HyperParam
    double epsilon; // Epsilon HyperParam
    double lr; // Learning Rate
    double decay; // Decay rate of lr
    int iterations; // Current training epoch
    bool correctBias; // Flag to determine if using bias correction
    OptimizationType optimizer; // Optimizer to Use
} OpParams;

/*
Initialize Adam Optimizer
*/
OpParams* init_adam(double beta_1, double beta_2, double epsilon, double lr, double decay);

/*
Free Adam Optimizer
*/
void free_adam(OpParams* adam);

/*
Run once before optimization
*/
void pre_update_params_adam(OpParams* adam);

/*
Run once after optimization
*/
void post_update_params_adam(OpParams* adam);

/*
Update dense layer parameters
*/
void update_dense_params_adam(OpParams* adam, layer_dense* layer);

/*
Update cnn layer parameters
*/
void update_cnn_params_adam();

/*
Update rnn layer parameters
*/
void update_rnn_params_adam();


#endif