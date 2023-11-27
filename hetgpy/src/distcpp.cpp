<%
cfg['compiler_args'] = ['-std=c++11']
cfg['include_dirs'] = ['../eigen']
setup_pybind11(cfg)
%>

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <Eigen/LU>

namespace py = pybind11;
Eigen::MatrixXd distcpp(Eigen::MatrixXd X1){
  int nr = X1.nrow();
  int nc = X1.ncol();
  Eigen::MatrixXd s(nr, nr);
  double tmp;

  for(int i = 1; i < nr; i++){
    double* ptrs = &s(0,i);
    double* ptrs2 = &s(i,0); //symmetric
    for(int j = 0; j < i; j++, ptrs++){
      const double* ptrX1 = (const double*) &X1(i,0);
      const double* ptrX2 = (const double*) &X1(j,0);
      for(int k = 0; k < nc; k++){
        tmp = (*ptrX1 - *ptrX2);
        *ptrs += tmp*tmp;
        ptrX1 += nr;
        ptrX2 += nr;
      }
      *ptrs2 = *ptrs;
      ptrs2 += nr;
    }
  }
  return s;
}

// // [[Rcpp::export]]
// Eigen::MatrixXd distcpp_v2(Eigen::MatrixXd X){
//   int nr = X.nrow();
//   int nc = X.ncol();
//   Eigen::MatrixXd s(nr, nr);
//   double tmp;
//
//   const double* ptrX1 = (const double*) &X(1,0);
//   const double* ptrX2 = (const double*) &X(0,0);
//   for(int i = 1; i < nr; i++){
//     double* ptrs = &s(0,i);
//     double* ptrs2 = &s(i,0); //symmetric
//     for(int j = 0; j < i; j++, ptrs++){
//
//       for(int k = 0; k < nc; k++){
//         tmp = (*ptrX1 - *ptrX2);
//         *ptrs += tmp*tmp;
//         ptrX1 += nr;
//         ptrX2 += nr;
//       }
//       *ptrs2 = *ptrs;
//       ptrs2 += nr;
//       ptrX1 -= nr*nc;
//       ptrX2 -= nr*nc - 1;
//     }
//     ptrX1++;
//     ptrX2 -= i;
//   }
//   return s;
// }



Eigen::MatrixXd distcpp_2(Eigen::MatrixXd X1, Eigen::MatrixXd X2){
  int nr1 = X1.nrow();
  int nr2 = X2.nrow();
  int dim = X1.ncol();
  Eigen::MatrixXd s(nr1, nr2);
  double tmp;

  double* ptrs = &s(0,0);
  const double* ptrX2 = (const double*) &X2(0,0);
  const double* ptrX1 = (const double*) &X1(0,0);
  for(int i = 0; i < nr2; i++){
    for(int j = 0; j < nr1; j++, ptrs++){
      for(int k = 0; k < dim; k++){
        tmp = (*ptrX1 - *ptrX2);
        *ptrs += tmp * tmp;
        ptrX1 += nr1;
        ptrX2 += nr2;
      }
      ptrX2 -= nr2*dim;
      ptrX1 -= nr1*dim - 1;
    }
    ptrX2++;
    ptrX1 -= nr1;
  }
  return s;
}

Eigen::MatrixXd distcppMaha(Eigen::MatrixXd X1, NumericVector m){
  int nr = X1.nrow();
  int nc = X1.ncol();
  Eigen::MatrixXd s(nr, nr);
  double tmp;

  for(int i = 1; i < nr; i++){
    double* ptrs = &s(0,i);
    double* ptrs2 = &s(i,0); //symmetric
    for(int j = 0; j < i; j++, ptrs++){
      const double* ptrX1 = (const double*) &X1(i,0);
      const double* ptrX2 = (const double*) &X1(j,0);
      double* ptrm = &m(0);
      for(int k = 0; k < nc; k++){
        tmp = (*ptrX1 - *ptrX2);
        *ptrs += tmp * tmp / *ptrm++;
        ptrX1 += nr;
        ptrX2 += nr;
      }
      *ptrs2 = *ptrs;
      ptrs2 += nr;
    }
  }
  return s;
}

Eigen::MatrixXd distcppMaha_2(Eigen::MatrixXd X1, Eigen::MatrixXd X2, NumericVector m){
  int nr1 = X1.nrow();
  int nr2 = X2.nrow();
  int dim = X1.ncol();
  Eigen::MatrixXd s(nr1, nr2);
  double tmp;

  double* ptrs = &s(0,0);
  const double* ptrX2 = (const double*) &X2(0,0);
  const double* ptrX1 = (const double*) &X1(0,0);
  double* ptrm = &m(0);
  for(int i = 0; i < nr2; i++){
    for(int j = 0; j < nr1; j++, ptrs++){
      for(int k = 0; k < dim; k++){
        tmp = (*ptrX1 - *ptrX2);
        *ptrs += tmp * tmp / *ptrm++;
        ptrX1 += nr1;
        ptrX2 += nr2;
      }
      ptrX2 -= nr2*dim;
      ptrX1 -= nr1*dim - 1;
      ptrm -= dim;
    }
    ptrX2++;
    ptrX1 -= nr1;
  }
  return s;
}

// [[Rcpp::export]]
Eigen::MatrixXd distance_cpp(Eigen::MatrixXd X1, Rcpp::Nullable<Rcpp::Eigen::MatrixXd> X2 = R_NilValue, Rcpp::Nullable<Rcpp::NumericVector> m = R_NilValue){
  Eigen::MatrixXd tmp;
  if(X2.isNotNull()){
    if(m.isNotNull()){
      tmp = distcppMaha_2(X1, (Eigen::MatrixXd) X2, (NumericVector) m);
    }else{
      tmp = distcpp_2(X1, (Eigen::MatrixXd) X2);
    }
  }else{
    if(m.isNotNull()){
      tmp = distcppMaha(X1, (NumericVector) m);
    }else{
      tmp = distcpp(X1);
    }
  }
  return(tmp);
}


// [[Rcpp::export]]
Eigen::MatrixXd partial_d_dist_dX_i1_i2(Eigen::MatrixXd X1, int i1, int i2){
  int nr = X1.nrow();
  Eigen::MatrixXd s(nr, nr);

  for(int i = 0; i < nr; i++){
    if(i == (i1 - 1))
      continue;
    s(i1 - 1, i) = s(i, i1 - 1) = -2 * (X1(i1-1, i2-1) - X1(i, i2-1));
  }
  return s;
}

// [[Rcpp::export]]
Eigen::MatrixXd partial_d_dist_dX1_i1_i2_X2(Eigen::MatrixXd X1, Eigen::MatrixXd X2, int i1, int i2){
  int nr = X2.nrow();
  Eigen::MatrixXd s(X1.nrow(), nr);

  for(int i = 0; i < nr; i++){
    s(i1 - 1, i) = -2 * (X1(i1-1, i2-1) - X2(i, i2-1));
  }
  return s;
}

