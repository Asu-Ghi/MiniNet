#include "linalg.h"
#include "layer_dense.h"
#include "relu.h"
#include "softmax.h"
#include "adam.h"

int main () {
    matrix test1;
    test1.rows = 2;
    test1.cols = 2;
    test1.data = (double*) malloc(sizeof(double) * test1.rows * test1.cols);
    double data1[4] = {1, 2, 3, 4};
    test1.data = data1;

    matrix pred1;
    pred1.rows = 2;
    pred1.cols = 5;
    pred1.data = (double*) malloc(sizeof(double) * pred1.rows * pred1.cols);
    double pred1data[10] = {0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
    pred1.data = pred1data;
    double beta_1 = 0.9;
    double beta_2 = 0.95;
    double epsilon = 1e-7;
    double lr = 0.05;
    double decay = 1e-5;

    layer_dense* layer1 = init_layer(2, 10);
    ReluParams* relu1 = init_relu();
    OpParams* adam1 = init_adam(beta_1, beta_2, epsilon, lr, decay);

    layer_dense* layer2 = init_layer(10, 5);
    SoftMaxParams* softmax2 = init_softmax();
    OpParams* adam2 = init_adam(beta_1, beta_2, epsilon, lr, decay);

    // Forward
    dense_forwards(&test1, layer1);
    relu_forwards(relu1, layer1->outputs);
    dense_forwards(relu1->outputs, layer2);
    softmax_forwards(softmax2, layer2->outputs);

    // Calculate Loss + Accuracy
    

    // Backward 
    softmax_backwards(softmax2, &pred1);
    dense_backwards(softmax2->dinputs, layer2);
    relu_backwards(relu1, layer2->dinputs);
    dense_backwards(relu1->dinputs, layer1);

    // Optimize Params
    pre_update_params_adam(adam1);
    update_dense_params_adam(adam1, layer1);
    post_update_params_adam(adam1);

    pre_update_params_adam(adam2);
    update_dense_params_adam(adam2, layer2);
    post_update_params_adam(adam2);
}