#include "loss.h"

Loss* init_loss(LossType loss_type) {
    Loss* loss_func = malloc(sizeof(Loss));
    if (loss_type == CATCROSSENTROPY) {
        loss_func->lossType = CATCROSSENTROPY;
        loss_func->loss = 0.0;
    }
    else if (loss_type == BINCROSSENTROPY) {
        loss_func->lossType = BINCROSSENTROPY;
        loss_func->loss = 0.0;
    }
    else if (loss_type == MSE) {
        loss_func->lossType = MSE;
        loss_func->loss = 0.0;
    }
    else if (loss_type == MAE) {
        loss_func->lossType = MAE;
        loss_func->loss = 0.0;
    }
    else {
        fprintf(stderr, "Error: Incorrect loss type provided in init categorical loss.\n");
        exit(1);
    }
    return loss_func;
}

void compute_loss (Loss* loss_func, matrix* X, matrix* Y) {
    // Dynamically allocate memory for X
    if (loss_func->X == NULL) {
        loss_func->X = allocate_matrix(X->rows, X->cols);
        memcpy(loss_func->X->data, X->data, X->rows * X->cols * sizeof(double));
    }

    // Calculate Loss
    if (loss_func->lossType == CATCROSSENTROPY) {
        calculate_catCE_loss(loss_func, Y);
    }
    else if (loss_func->lossType == BINCROSSENTROPY) {
        calculate_binCE_loss(loss_func, Y);
    }
    else if (loss_func->lossType == MSE) {
        calculate_MSE_loss(loss_func, Y);
    }
    else if (loss_func->lossType == MAE) {
        calculate_MAE_loss(loss_func, Y);
    }

    // Free X
    free_matrix(loss_func->X);
}

void calculate_catCE_loss(Loss* loss_func, matrix* Y) {

    // initialize losses data.
    double losses = 0.0;

    // check if one hot is the correct size
    if (loss_func->X->cols != Y->cols) {
        fprintf(stderr, "Error: Dimension 2 for one hot vectors and predictions do not match in calculate catCE loss.\n");
        exit(1);
    }

    // iterate over every vector in the prediction batch
    for (int i = 0; i < loss_func->X->rows; i++) {

        // find true class in one hot vector
        int true_class = -1;
        for (int j = 0; j < Y->cols; j++) {
            if (Y->data[i * Y->cols + j] == 1.0) {
                true_class = j;
                break;
            }
        }

        // error handling if no true class is found
        if(true_class == -1) {
            fprintf(stderr, "Error: No true class found in one hot vectors in calculate cat CE loss. \n");
            exit(1);
        }

        // get predicted sample in question with relation to true class
        double predicted_sample = loss_func->X->data[i * loss_func->X->rows + true_class];

        // clip value so we never calculate log(0)
        if(predicted_sample < 1e-15) {
            predicted_sample = 1e-15;
        }
        
        // calcuale -log loss for the sample in question and append to loss matrix
        double loss = -log(predicted_sample);
        losses += loss;
    }

    // Update Loss
    loss_func->loss = (losses / Y->rows);    
}

void calculate_binCE_loss(Loss* loss_func, matrix* Y) {

    // Check for dimension compatibility
    if (loss_func->X->rows != Y->rows || Y->cols != 1) {
        fprintf(stderr, "Error: Dimensionality Mismatch between prediction and true label dimensions in calculate binary CE loss.\n");
        exit(1);
    }

    double total_loss = 0.0; 

    for (int i = 0; i < loss_func->X->rows; i++) {
        double sample_loss = 0.0;
        for (int j = 0; j < loss_func->X->cols; j++) {
            double y_hat = loss_func->X->data[i * 
                        loss_func->X->cols + j]; 

            double y = Y->data[i * Y->cols + j]; 

            // clip value so we never calculate log(0)
            if(y_hat < 1e-15) {
                y_hat = 1e-15;
            } 
            sample_loss -= y * log(y_hat) + (1.0 - y) * log(1.0 - y_hat);  // Binary CE formula
        }
        total_loss += sample_loss;
    }
    
    // Update loss
    loss_func->loss = (total_loss / loss_func->X->rows);    
}

void calculate_MSE_loss(Loss* loss_func, matrix* Y) {
    printf("Not supported yet\n");
}

void calculate_MAE_loss(Loss* loss_func, matrix* Y) {
    printf("Not supported yet\n");
}