#ifndef CONFIG_H
#define CONFIG_H

//#define CUSTOM_PRIORS
//#define CUSTOM_INITIALS


//////////////////////////////////////////////////////////
//
// Prior Parameters
//
//////////////////////////////////////////////////////////

#ifdef CUSTOM_PRIORS

#define GAMMA_PRIOR 1.0

#define ALPHA_A_PRIOR 1.0

#define ALPHA_B_PRIOR 1.0

#define ALPHA_PRIOR 1.0

#define KAPPA_PRIOR 1.0

#define NU_PRIOR 4.0

#define M_PRIOR { 0, 0 }

#define S_PRIOR { { 1, 0 }, \
				  { 0, 1 }  }

#else

#define GAMMA_PRIOR 1.0

#define ALPHA_A_PRIOR 1.0

#define ALPHA_B_PRIOR 1.0

#define ALPHA_PRIOR 1.0

#define KAPPA_PRIOR 1.0

#define NU_PRIOR 2.0 + dim

#define M_PRIOR arma::zeros<arma::vec>(dim)

#define S_PRIOR arma::eye(dim, dim)

#endif 

//////////////////////////////////////////////////////////
//
// Initial Values
//
//////////////////////////////////////////////////////////


#endif 

