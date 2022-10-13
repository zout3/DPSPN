#pragma once

#define DAT_DIR std::string("D:\\project\\dat\\")

//////////////////////////////////////////////////////////
//
// Data Dimension
//
//////////////////////////////////////////////////////////

#define DIM_d1 2  // dimension of fixed identity matrix in covariance matrix

#define DIM_p 2  // dimension of directional variable 

//////////////////////////////////////////////////////////
//
// Sampling Setup
//
//////////////////////////////////////////////////////////

#define N_ITER 1000

#define N_BURN 500

#define N_THIN 5

#define UPDATE_ALPHA 0

//////////////////////////////////////////////////////////
//
// Prior Parameters
//
//////////////////////////////////////////////////////////

#define GAMMA_PRIOR 1.0

#define ALPHA_A_PRIOR 1.0

#define ALPHA_B_PRIOR 1.0

#define ALPHA_PRIOR 1.0

#define KAPPA_PRIOR 1.0

#define NU_PRIOR 2.0 + dim

#define M_PRIOR arma::zeros<arma::vec>(dim)

#define S_PRIOR arma::eye(dim, dim)
