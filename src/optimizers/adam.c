#include "adam.h"


OpParams* init_adam(double beta_1, double beta_2, double epsilon,
                    double lr, double decay) {

    OpParams* adam = malloc(sizeof(OpParams)); 
    adam->w_momentums = NULL;
    adam->w_cache = NULL;
    adam->b_momentums = NULL;
    adam->b_cache = NULL;
    adam->beta_1 = beta_1;
    adam->beta_2 = beta_2;
    adam->epsilon = epsilon;
    adam->lr = lr;
    adam->decay = decay;
    adam->iterations = 0;
    adam->correctBias = true;
    return adam;
}

void free_adam(OpParams* adam) {
    if (adam->w_momentums != NULL) {
        free_matrix(adam->w_momentums);
    }
    if (adam->w_cache != NULL) {
        free_matrix(adam->w_cache);
    }
    if (adam->b_momentums != NULL) {
        free_matrix(adam->b_momentums);
    }
    if (adam->b_cache != NULL) {
        free_matrix(adam->b_cache);
    }
}

void pre_update_params_adam(OpParams* adam) {
    if (adam->decay > 0.0) {
        adam->lr = adam->lr * (1.0 / (1 + adam->decay * adam->iterations));
    }
}

void post_update_params_adam(OpParams* adam) {
    adam->iterations += 1;
}

void update_dense_params_adam(OpParams* adam, layer_dense* layer) {
    // Allocate adam struct memory dynamically
    if (adam->w_momentums == NULL) {
        adam->w_momentums = allocate_matrix(layer->weights->rows, layer->weights->cols);
    }
    if (adam->w_cache == NULL) {
        adam->w_cache = allocate_matrix(layer->weights->rows, layer->weights->cols);
    }
    if (adam->b_momentums == NULL) {
        adam->b_momentums = allocate_matrix(layer->biases->rows, layer->biases->cols);
    }
    if (adam->b_cache == NULL) {
        adam->b_cache = allocate_matrix(layer->biases->rows, layer->biases->cols);
    }

    // Update layer parameters
    #ifdef ENABLE_PARALLEL 
    // Weights
    #pragma omp for schedule(dynamic)
    for (int i = 0; i < layer->weights->rows * layer->weights->cols; i++) {

        // Update Momentum
        adam->w_momentums->data[i] = adam->beta_1 * adam->w_momentums->data[i] + (1.0 - adam->beta_1) * layer->dweights->data[i];
        
        // Correct Momentum
        if (adam->correctBias) {
            long double beta_1_pow = pow(adam->beta_1, adam->iterations + 1);
            adam->w_momentums->data[i] = adam->w_momentums->data[i] / (1.0 - beta_1_pow); // Bias correction for weights momentum
        }

        // Update cache 
        adam->w_cache->data[i] = adam->beta_2 * adam->w_cache->data[i] + (1.0 - adam->beta_2) * (layer->dweights->data[i] * layer->dweights->data[i]);
        
        // Correct cache
        if (adam->correctBias) {
            long double beta_2_pow = pow(adam->beta_2, adam->iterations + 1);
            adam->w_cache->data[i] = adam->w_cache->data[i] / (1.0 - beta_2_pow); // Bias correction for weight cache
        }

        // Update Weights using corrected moments and cache
        layer->weights->data[i] -= (adam->lr) * adam->w_momentums->data[i] / (sqrt(adam->w_cache->data[i]) + adam->epsilon);
    }
    
    // Biases
    #pragma omp for schedule(dynamic)
    for (int i = 0; i < layer->biases->rows * layer->biases->cols; i++) {
        // Update Momentum
        adam->b_momentums->data[i] = adam->beta_1 * adam->b_momentums->data[i] + (1.0 - adam->beta_1) * layer->dbiases->data[i];
        
        // Correct Momentum
        if (adam->correctBias) {
            long double beta_1_pow = pow(adam->beta_1, adam->iterations+1);
            adam->b_momentums->data[i] = adam->b_momentums->data[i] / (1.0 - beta_1_pow); // Bias correction for bias momentum
        }
        
        // Update cache 
        adam->b_cache->data[i] = adam->beta_2 * adam->b_cache->data[i] + (1.0 - adam->beta_2) * layer->dbiases->data[i] * layer->dbiases->data[i];
        
        // Correct cache
        if (adam->correctBias) {
            long double beta_2_pow = pow(adam->beta_2, adam->iterations+1);
            adam->b_cache->data[i] = adam->b_cache->data[i] / (1.0 - beta_2_pow); // Bias correction for bias cache
        }

        // Update Biases using corrected moments and cache
        layer->biases->data[i] -= (adam->lr) * adam->b_momentums->data[i] / (sqrt(adam->b_cache->data[i]) + adam->epsilon);
    }

    # else 
    // Weights
    for (int i = 0; i < layer->weights->rows * layer->weights->cols; i++) {

        // Update Momentum
        adam->w_momentums->data[i] = adam->beta_1 * adam->w_momentums->data[i] + (1.0 - adam->beta_1) * layer->dweights->data[i];
        
        // Correct Momentum
        if (adam->correctBias) {
            long double beta_1_pow = pow(adam->beta_1, adam->iterations + 1);
            adam->w_momentums->data[i] = adam->w_momentums->data[i] / (1.0 - beta_1_pow); // Bias correction for weights momentum
        }

        // Update cache 
        adam->w_cache->data[i] = adam->beta_2 * adam->w_cache->data[i] + (1.0 - adam->beta_2) * (layer->dweights->data[i] * layer->dweights->data[i]);
        
        // Correct cache
        if (adam->correctBias) {
            long double beta_2_pow = pow(adam->beta_2, adam->iterations + 1);
            adam->w_cache->data[i] = adam->w_cache->data[i] / (1.0 - beta_2_pow); // Bias correction for weight cache
        }

        // Update Weights using corrected moments and cache
        layer->weights->data[i] -= (adam->lr) * adam->w_momentums->data[i] / (sqrt(adam->w_cache->data[i]) + adam->epsilon);
    }
    
    // Biases
    for (int i = 0; i < layer->biases->rows * layer->biases->cols; i++) {
        // Update Momentum
        adam->b_momentums->data[i] = adam->beta_1 * adam->b_momentums->data[i] + (1.0 - adam->beta_1) * layer->dbiases->data[i];
        
        // Correct Momentum
        if (adam->correctBias) {
            long double beta_1_pow = pow(adam->beta_1, adam->iterations+1);
            adam->b_momentums->data[i] = adam->b_momentums->data[i] / (1.0 - beta_1_pow); // Bias correction for bias momentum
        }
        
        // Update cache 
        adam->b_cache->data[i] = adam->beta_2 * adam->b_cache->data[i] + (1.0 - adam->beta_2) * layer->dbiases->data[i] * layer->dbiases->data[i];
        
        // Correct cache
        if (adam->correctBias) {
            long double beta_2_pow = pow(adam->beta_2, adam->iterations+1);
            adam->b_cache->data[i] = adam->b_cache->data[i] / (1.0 - beta_2_pow); // Bias correction for bias cache
        }

        // Update Biases using corrected moments and cache
        layer->biases->data[i] -= (adam->lr) * adam->b_momentums->data[i] / (sqrt(adam->b_cache->data[i]) + adam->epsilon);
    }

    #endif 
}
