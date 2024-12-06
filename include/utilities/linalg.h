#ifndef LINALG_H
#define LINALG_H
#include "global.h"


//////////////////////////////////////////////////// HELPER FUNCTIONS //////////////////////////////////////////////////////////////

/*
Allocates memory on the heap for a matrix object.
Checks memory allocation.
*/
matrix* allocate_matrix(int dim1, int dim2);

/*
Frees matrix struct. Checks for dangling pointers.
*/
void free_matrix(matrix* M);

/*
Shallow copies a select portion of the matrix src
*/
void shallow_cpy_matrix(matrix* src, matrix* dest, int start_row, int num_rows);

/*
Prints a matrix object and its dimensions.
*/
void print_matrix(matrix* M);

/*
Fills a Matrix M with a value
*/
void fill_matrix(matrix* M, double val);

//////////////////////////////////////////////////// LIN ALG FUNCTIONS //////////////////////////////////////////////////////////////

/*
Returns a matrix object. 
Transposes w, swaps dimension indicators in the matrix object.
Allocates memory on the heap for the return matrix.
*/
matrix* transpose_matrix(matrix* w); 

/*
Returns a matrix object. 
Includes dimensionality checks.
Allocates memory on the heap for the return matrix.
*/
matrix* matrix_mult(matrix* w, matrix* v);

/*
Returns a matrix object. 
Includes dimensionality checks.
Allocates memory on the heap for the return matrix
*/
matrix* element_matrix_mult(matrix* w, matrix* v);

/*
Returns matrix object
Includes dimensionality checks.
Allocates memory on the heap for the return matrix
*/
matrix* matrix_sum(matrix* w, matrix* v);

/*
Returns a matrix object
Allocates memory on the heap for the return matrix
*/
matrix* matrix_scalar_sum(matrix* w, double s, bool useAbs);

/*
Returns average value of the matrix.
*/
double matrix_mean(matrix* w);


#endif