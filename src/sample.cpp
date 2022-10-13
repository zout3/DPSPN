#include "sample.h"

//Modified sample_main from RcppArmadilloExtensions/sample.h

void SampleNoReplace(arma::uvec& index, int nOrig, int size) {
    int ii, jj;
    arma::uvec sub(nOrig);
    for (ii = 0; ii < nOrig; ii++) {
        sub(ii) = ii;
    }
    for (ii = 0; ii < size; ii++) {
        jj = nOrig * arma::randu();
        index(ii) = sub(jj);
        // replace sampled element with last, decrement
        sub(jj) = sub(--nOrig);
    }
}

void ProbSampleNoReplace(arma::uvec& index, int nOrig, int size, arma::vec& prob) {
    int ii, jj, kk;
    int nOrig_1 = nOrig - 1;
    double rT, mass, totalmass = 1.0;
    arma::uvec perm = arma::sort_index(prob, "descend"); //descending sort of index
    prob = arma::sort(prob, "descend");  // descending sort of prob
    // compute the sample 
    for (ii = 0; ii < size; ii++, nOrig_1--) {
        rT = totalmass * arma::randu();
        mass = 0;
        for (jj = 0; jj < nOrig_1; jj++) {
            mass += prob[jj];
            if (rT <= mass)
                break;
        }
        index[ii] = perm[jj];
        totalmass -= prob[jj];
        for (kk = jj; kk < nOrig_1; kk++) {
            prob[kk] = prob[kk+1];
            perm[kk] = perm[kk+1];
        }
    }
}


arma::uvec sample(const arma::uvec& x, const int size, const arma::vec& prob) {

    int ii, jj;
    int nOrig = x.size();
    int probsize = prob.n_elem;

    arma::uvec ret(size);

    arma::uvec index(size);
    if (probsize == 0) {
        SampleNoReplace(index, nOrig, size);
    }
    else {
        arma::vec fprob = prob;
        ProbSampleNoReplace(index, nOrig, size, fprob);
    }

    for (ii = 0; ii < size; ii++) {
        jj = index(ii);

        ret[ii] = x[jj];
    }
    return(ret);
}


arma::uvec sample(const arma::uvec& x, const int size) {
    // Creates a zero-size vector in arma (cannot directly call arma::vec(0))
    const arma::vec prob = arma::zeros<arma::vec>(0);
    return sample(x, size, prob);
}