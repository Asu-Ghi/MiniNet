#ifndef LOSS_H
#define LOSS_H
#include "linalg.h"

/*
Initializes loss struct for loss
*/
Loss* init_loss(LossType loss_type);

/*
Computes loss (calls func)
*/
void compute_loss (Loss* loss_func, matrix* Y);

/*
Computes loss for Categorical Cross Entropy
*/
void calculate_catCE_loss(Loss* loss_func, matrix* Y);

/*
Computes loss for Binary Cross Entropy
*/
void calculate_binCE_loss(Loss* loss_func, matrix* Y);

/*
Computes loss for Mean Squared Error
*/
void calculate_MSE_loss(Loss* loss_func, matrix* Y);

/*
Computes loss for Mean Absolute Error
*/
void calculate_MAE_loss(Loss* loss_func, matrix* Y);

#endif