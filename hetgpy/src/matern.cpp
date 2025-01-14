// cppimport
/*
<%
cfg['compiler_args'] = ['-std=c++17']
cfg['include_dirs'] = ['../../eigen']
setup_pybind11(cfg)
%>
*/
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <Eigen/LU>

namespace py = pybind11;

// [[Rcpp::export]]
Eigen::MatrixXd matern5_2_1args(Eigen::MatrixXd X1){
  int nr = X1.rows();
  int nc = X1.cols();
  
  Eigen::MatrixXd s = Eigen::MatrixXd::Ones(nr, nr);
  Eigen::MatrixXd r = Eigen::MatrixXd::Zero(nr, nr);
  
  double tmp;
  
  // First compute polynomial term and distance
  const double* ptrX1 = (const double*) &X1(1,0);
  const double* ptrX2 = (const double*) &X1(0,0);
  double* ptrs = &s(0,1);
  double* ptrr = &r(0,1);
  
  for(int i = 1; i < nr; i++, ptrX1++){
    for(int j = 0; j < i; j++, ptrs++, ptrr++){
      for(int k = 0; k < nc; k++){
        tmp = sqrt(5.) * std::abs(*ptrX1 - *ptrX2);
        *ptrs *= (1 + tmp + tmp * tmp /3.);
        *ptrr -= tmp;
        ptrX1 += nr;
        ptrX2 += nr;
      }
      ptrX1 -= nr*nc;
      ptrX2 -= nr*nc - 1;
    }
    
    ptrX2 -= i;
    ptrs += (nr - i);
    ptrr += (nr - i);
  }
  
  // Now multiply by exponential part
  double* ptrs2 = &s(1,0); //symmetric
  ptrs = &s(0,1);
  ptrr = &r(0,1);
  for(int i = 1; i < nr; i++){
    for(int j = 0; j < i; j++, ptrs++, ptrr++){
      *ptrs *= exp(*ptrr);
      *ptrs2 = *ptrs;
      ptrs2 += nr;
    }
    ptrs += (nr - i);
    ptrr += (nr - i);
    ptrs2 += 1 - i*nr;
  }
  
  return s;
}


// [[Rcpp::export]]
Eigen::MatrixXd d_matern5_2_1args_theta_k_iso(Eigen::MatrixXd X1, double theta){
  int nr = X1.rows();
  int nc = X1.cols();
  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(nr, nr);
  double tmp;
  
  const double* ptrX1 = (const double*) &X1(1,0);
  const double* ptrX2 = (const double*) &X1(0,0);
  double* ptrs = &s(0,1);
  double* ptrs2 = &s(1,0); //symmetric
  
  for(int i = 1; i < nr; i++, ptrX1++){
    for(int j = 0; j < i; j++, ptrs++){
      for(int k = 0; k < nc; k++){
        tmp = std::abs(*ptrX1 - *ptrX2) / theta;
        *ptrs -= ((10./3. - 5.) * tmp - 5 * sqrt(5.)/3. * tmp * tmp) / (1 + sqrt(5.) * tmp + 5./3. * tmp * tmp) * tmp/theta;
        ptrX1 += nr;
        ptrX2 += nr;
      }
      *ptrs2 = *ptrs;
      ptrs2 += nr;
      ptrX1 -= nr*nc;
      ptrX2 -= nr*nc - 1;
    }
    
    ptrX2 -= i;
    ptrs += (nr - i);
    ptrs2 += 1 - i*nr;
  }
  return s;
}

// [[Rcpp::export]]
Eigen::MatrixXd d_matern5_2_1args_theta_k(Eigen::MatrixXd X1, double theta){
  // X1 has just one column here
  int nr = X1.rows();
  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(nr, nr);
  double tmp;
  
  const double* ptrX1 = (const double*) &X1(1,0);
  const double* ptrX2 = (const double*) &X1(0,0);
  double* ptrs = &s(0,1);
  double* ptrs2 = &s(1,0); //symmetric
  
  for(int i = 1; i < nr; i++, ptrX1++){
    for(int j = 0; j < i; j++, ptrs++){
      tmp = std::abs(*ptrX1 - *ptrX2) / theta;
      
      *ptrs -= ((10./3. - 5.) * tmp - 5 * sqrt(5.)/3. * tmp * tmp) / (1 + sqrt(5.) * tmp + 5./3. * tmp * tmp) * tmp/theta;
      
      *ptrs2 = *ptrs;
      ptrs2 += nr;
      ptrX2 ++;
    }
    
    ptrX2 -= i;
    ptrs += (nr - i);
    ptrs2 += 1 - i*nr;
  }
  return s;
}

// [[Rcpp::export]]
Eigen::MatrixXd d_matern5_2_1args_kthetag(Eigen::MatrixXd X1, double kt){
  int nr = X1.rows();
  int nc = X1.cols();
  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(nr, nr);
  double tmp;
  
  const double* ptrX1 = (const double*) &X1(1,0);
  const double* ptrX2 = (const double*) &X1(0,0);
  double* ptrs = &s(0,1);
  double* ptrs2 = &s(1,0); //symmetric
  
  for(int i = 1; i < nr; i++, ptrX1++){
    for(int j = 0; j < i; j++, ptrs++){
      for(int k = 0; k < nc; k++){
        tmp = std::abs(*ptrX1 - *ptrX2) / kt;
        *ptrs -= ((10./3. - 5.) * tmp - 5 * sqrt(5.)/3. * tmp * tmp) / (1 + sqrt(5.) * tmp + 5./3. * tmp * tmp) * tmp/kt;
        ptrX1 += nr;
        ptrX2 += nr;
      }
      *ptrs2 = *ptrs;
      ptrs2 += nr;
      ptrX1 -= nr*nc;
      ptrX2 -= nr*nc - 1;
    }
    
    ptrX2 -= i;
    ptrs += (nr - i);
    ptrs2 += 1 - i*nr;
  }
  return s;
}

// [[Rcpp::export]]
Eigen::MatrixXd matern5_2_2args(Eigen::MatrixXd X1, Eigen::MatrixXd X2){
  int nr1 = X1.rows();
  int nr2 = X2.rows();
  int dim = X1.cols();
  
  Eigen::MatrixXd s = Eigen::MatrixXd::Ones(nr1, nr2);
  Eigen::MatrixXd r = Eigen::MatrixXd::Zero(nr1, nr2);
  double tmp;
  
  double* ptrs = &s(0,0);
  double* ptrr = &r(0,0);
  const double* ptrX2 = (const double*) &X2(0,0);
  const double* ptrX1 = (const double*) &X1(0,0);
  
  // Polynomial part
  for(int i = 0; i < nr2; i++){
    for(int j = 0; j < nr1; j++, ptrs++, ptrr++){
      for(int k = 0; k < dim; k++){
        tmp = sqrt(5.) * std::abs(*ptrX1 - *ptrX2);
        *ptrs *= (1 + tmp + tmp * tmp / 3.);
        *ptrr -= tmp;
        ptrX1 += nr1;
        ptrX2 += nr2;
      }
      ptrX2 -= nr2*dim;
      ptrX1 -= nr1*dim - 1;
      
    }
    ptrX2++;
    ptrX1 -= nr1;
  }
  
  ptrs = &s(0,0);
  ptrr = &r(0,0);
  
  // Exponential part
  for(int i = 0; i < nr2; i++){
    for(int j = 0; j < nr1; j++, ptrs++, ptrr++){
      *ptrs *= exp(*ptrr);
    }
  }
  
  return s;
}


// [[Rcpp::export]]
Eigen::MatrixXd d_matern5_2_2args_theta_k_iso(Eigen::MatrixXd X1, Eigen::MatrixXd X2, double theta){
  int nr1 = X1.rows();
  int nr2 = X2.rows();
  int dim = X1.cols();
  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(nr1, nr2);
  double tmp;
  
  double* ptrs = &s(0,0);
  const double* ptrX2 = (const double*) &X2(0,0);
  const double* ptrX1 = (const double*) &X1(0,0);
  
  for(int i = 0; i < nr2; i++){
    for(int j = 0; j < nr1; j++, ptrs++){
      for(int k = 0; k < dim; k++){
        tmp = std::abs(*ptrX1 - *ptrX2) / theta;
        *ptrs -= ((10./3. - 5.) * tmp - 5 * sqrt(5.)/3. * tmp * tmp) / (1 + sqrt(5.) * tmp + 5./3. * tmp * tmp) * tmp / theta;
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


// [[Rcpp::export]]
Eigen::MatrixXd d_matern5_2_2args_kthetag(Eigen::MatrixXd X1, Eigen::MatrixXd X2, double kt){
  int nr1 = X1.rows();
  int nr2 = X2.rows();
  int dim = X1.cols();
  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(nr1, nr2);
  double tmp;
  
  double* ptrs = &s(0,0);
  const double* ptrX2 = (const double*) &X2(0,0);
  const double* ptrX1 = (const double*) &X1(0,0);
  for(int i = 0; i < nr2; i++){
    for(int j = 0; j < nr1; j++, ptrs++){
      for(int k = 0; k < dim; k++){
        tmp = std::abs(*ptrX1 - *ptrX2) / kt;
        *ptrs -= ((10./3. - 5.) * tmp - 5 * sqrt(5.)/3. * tmp * tmp) / (1 + sqrt(5.) * tmp + 5./3. * tmp * tmp) * tmp/kt;
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

// [[Rcpp::export]]
Eigen::MatrixXd partial_d_dist_abs_dX_i1_i2(Eigen::MatrixXd X1, int i1, int i2){
  int nr = X1.rows();
  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(nr, nr);
  double tmp;
  
  for(int i = 0; i < nr; i++){
    if(i == (i1 - 1))
      continue;
    tmp = (X1(i1 - 1, i2 - 1) - X1(i, i2 - 1)) ;
    if(tmp > 0){
      s(i1 - 1, i) = s(i, i1 - 1) = ((10./3. - 5.) * tmp - 5 * sqrt(5.)/3. * tmp * tmp) / (1. + sqrt(5.) * tmp + 5./3. * tmp * tmp);
    }else{
      if(tmp == 0){
        s(i1 - 1, i) = s(i, i1 - 1) = 0;
      }else{
        tmp = std::abs(tmp);
        s(i1 - 1, i) = s(i, i1 - 1) = -((10./3. - 5.) * tmp - 5. * sqrt(5.)/3. * tmp * tmp) / ((1. + sqrt(5.) * tmp + 5./3. * tmp * tmp));
      }
    }
  }
  return s;
}

// [[Rcpp::export]]
Eigen::MatrixXd partial_d_dist_abs_dX1_i1_i2_X2(Eigen::MatrixXd X1, Eigen::MatrixXd X2, int i1, int i2){
  int nr = X2.rows();
  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(X1.rows(), nr);
  double tmp;
  
  for(int i = 0; i < nr; i++){
    tmp = X1(i1-1, i2-1) - X2(i, i2-1);
    if(tmp > 0){
      s(i1 - 1, i) = ((10./3. - 5.) * tmp - 5 * sqrt(5.)/3. * tmp * tmp) / (1. + sqrt(5.) * tmp + 5./3. * tmp * tmp);
    }else{
      if(tmp == 0){
        s(i1 - 1, i) = 0;
      }else{
        tmp = std::abs(tmp);
        s(i1 - 1, i) = -((10./3. - 5.) * tmp - 5. * sqrt(5.)/3. * tmp * tmp) / ((1. + sqrt(5.) * tmp + 5./3. * tmp * tmp));
      }
    }
  }
  return s;
}



// [[Rcpp::export]]
Eigen::MatrixXd matern3_2_1args(Eigen::MatrixXd X1){
  int nr = X1.rows();
  int nc = X1.cols();
  
  Eigen::MatrixXd s = Eigen::MatrixXd::Ones(nr, nr);
  Eigen::MatrixXd r = Eigen::MatrixXd::Zero(nr, nr);
  double tmp;
  
  // First compute polynomial term and distance
  const double* ptrX1 = (const double*) &X1(1,0);
  const double* ptrX2 = (const double*) &X1(0,0);
  double* ptrs = &s(0,1);
  double* ptrr = &r(0,1);
  
  for(int i = 1; i < nr; i++, ptrX1++){
    for(int j = 0; j < i; j++, ptrs++, ptrr++){
      for(int k = 0; k < nc; k++){
        tmp = sqrt(3.) * std::abs(*ptrX1 - *ptrX2);
        *ptrs *= (1 + tmp);
        *ptrr -= tmp;
        ptrX1 += nr;
        ptrX2 += nr;
      }
      ptrX1 -= nr*nc;
      ptrX2 -= nr*nc - 1;
    }
    
    ptrX2 -= i;
    ptrs += (nr - i);
    ptrr += (nr - i);
  }
  
  // Now multiply by exponential part
  double* ptrs2 = &s(1,0); //symmetric
  ptrs = &s(0,1);
  ptrr = &r(0,1);
  for(int i = 1; i < nr; i++){
    for(int j = 0; j < i; j++, ptrs++, ptrr++){
      *ptrs *= exp(*ptrr);
      *ptrs2 = *ptrs;
      ptrs2 += nr;
    }
    ptrs += (nr - i);
    ptrr += (nr - i);
    ptrs2 += 1 - i*nr;
  }
  
  return s;
}


// [[Rcpp::export]]
Eigen::MatrixXd d_matern3_2_1args_theta_k_iso(Eigen::MatrixXd X1, double theta){
  int nr = X1.rows();
  int nc = X1.cols();
  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(nr, nr);
  double tmp;
  
  const double* ptrX1 = (const double*) &X1(1,0);
  const double* ptrX2 = (const double*) &X1(0,0);
  double* ptrs = &s(0,1);
  double* ptrs2 = &s(1,0); //symmetric
  
  for(int i = 1; i < nr; i++, ptrX1++){
    for(int j = 0; j < i; j++, ptrs++){
      for(int k = 0; k < nc; k++){
        tmp = std::abs(*ptrX1 - *ptrX2) / theta;
        *ptrs = 3*tmp / (1 + sqrt(3.) * tmp) * tmp / theta;
        ptrX1 += nr;
        ptrX2 += nr;
      }
      *ptrs2 = *ptrs;
      ptrs2 += nr;
      ptrX1 -= nr*nc;
      ptrX2 -= nr*nc - 1;
    }
    
    ptrX2 -= i;
    ptrs += (nr - i);
    ptrs2 += 1 - i*nr;
  }
  return s;
}

// [[Rcpp::export]]
Eigen::MatrixXd d_matern3_2_1args_theta_k(Eigen::MatrixXd X1, double theta){
  // X1 has one column
  int nr = X1.rows();
  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(nr, nr);
  double tmp;
  
  const double* ptrX1 = (const double*) &X1(1,0);
  const double* ptrX2 = (const double*) &X1(0,0);
  double* ptrs = &s(0,1);
  double* ptrs2 = &s(1,0); //symmetric
  
  for(int i = 1; i < nr; i++, ptrX1++){
    for(int j = 0; j < i; j++, ptrs++){
      tmp = std::abs(*ptrX1 - *ptrX2) / theta;
      *ptrs = 3*tmp / (1 + sqrt(3.) * tmp) * tmp / theta;
      *ptrs2 = *ptrs;
      ptrs2 += nr;
      ptrX2++;
    }
    
    ptrX2 -= i;
    ptrs += (nr - i);
    ptrs2 += 1 - i*nr;
  }
  return s;
}

// [[Rcpp::export]]
Eigen::MatrixXd d_matern3_2_1args_kthetag(Eigen::MatrixXd X1, double kt){
  int nr = X1.rows();
  int nc = X1.cols();
  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(nr, nr);
  double tmp;
  
  const double* ptrX1 = (const double*) &X1(1,0);
  const double* ptrX2 = (const double*) &X1(0,0);
  double* ptrs = &s(0,1);
  double* ptrs2 = &s(1,0); //symmetric
  
  for(int i = 1; i < nr; i++, ptrX1++){
    for(int j = 0; j < i; j++, ptrs++){
      for(int k = 0; k < nc; k++){
        tmp = std::abs(*ptrX1 - *ptrX2) / kt;
        *ptrs -= 3*tmp / (1 + sqrt(3.) * tmp) * tmp/kt;
        ptrX1 += nr;
        ptrX2 += nr;
      }
      *ptrs2 = *ptrs;
      ptrs2 += nr;
      ptrX1 -= nr*nc;
      ptrX2 -= nr*nc - 1;
    }
    
    ptrX2 -= i;
    ptrs += (nr - i);
    ptrs2 += 1 - i*nr;
  }
  return s;
}

// [[Rcpp::export]]
Eigen::MatrixXd matern3_2_2args(Eigen::MatrixXd X1, Eigen::MatrixXd X2){
  int nr1 = X1.rows();
  int nr2 = X2.rows();
  int dim = X1.cols();
  
  Eigen::MatrixXd s = Eigen::MatrixXd::Ones(nr1, nr2);
  Eigen::MatrixXd r = Eigen::MatrixXd::Zero(nr1, nr2);
  double tmp;
  
  double* ptrs = &s(0,0);
  double* ptrr = &r(0,0);
  const double* ptrX2 = (const double*) &X2(0,0);
  const double* ptrX1 = (const double*) &X1(0,0);
  
  // Polynomial part
  for(int i = 0; i < nr2; i++){
    for(int j = 0; j < nr1; j++, ptrs++, ptrr++){
      for(int k = 0; k < dim; k++){
        tmp = sqrt(3.) * std::abs(*ptrX1 - *ptrX2);
        *ptrs *= (1 + tmp);
        *ptrr -= tmp;
        ptrX1 += nr1;
        ptrX2 += nr2;
      }
      ptrX2 -= nr2*dim;
      ptrX1 -= nr1*dim - 1;
      
    }
    ptrX2++;
    ptrX1 -= nr1;
  }
  
  ptrs = &s(0,0);
  ptrr = &r(0,0);
  
  // Exponential part
  for(int i = 0; i < nr2; i++){
    for(int j = 0; j < nr1; j++, ptrs++, ptrr++){
      *ptrs *= exp(*ptrr);
    }
  }
  
  return s;
}


// [[Rcpp::export]]
Eigen::MatrixXd d_matern3_2_2args_theta_k_iso(Eigen::MatrixXd X1, Eigen::MatrixXd X2, double theta){
  int nr1 = X1.rows();
  int nr2 = X2.rows();
  int dim = X1.cols();
  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(nr1, nr2);
  double tmp;
  
  double* ptrs = &s(0,0);
  const double* ptrX2 = (const double*) &X2(0,0);
  const double* ptrX1 = (const double*) &X1(0,0);
  
  for(int i = 0; i < nr2; i++){
    for(int j = 0; j < nr1; j++, ptrs++){
      for(int k = 0; k < dim; k++){
        tmp = std::abs(*ptrX1 - *ptrX2) / theta;
        *ptrs = 3*tmp / (1 + sqrt(3.) * tmp) * tmp / theta;
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


// [[Rcpp::export]]
Eigen::MatrixXd d_matern3_2_2args_kthetag(Eigen::MatrixXd X1, Eigen::MatrixXd X2, double kt){
  int nr1 = X1.rows();
  int nr2 = X2.rows();
  int dim = X1.cols();
  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(nr1, nr2);
  double tmp;
  
  double* ptrs = &s(0,0);
  const double* ptrX2 = (const double*) &X2(0,0);
  const double* ptrX1 = (const double*) &X1(0,0);
  for(int i = 0; i < nr2; i++){
    for(int j = 0; j < nr1; j++, ptrs++){
      for(int k = 0; k < dim; k++){
        tmp = std::abs(*ptrX1 - *ptrX2) / kt;
        *ptrs = 3*tmp / (1 + sqrt(3.) * tmp) * tmp/kt;
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

// [[Rcpp::export]]
Eigen::MatrixXd partial_d_dist_abs_dX_i1_i2_m32(Eigen::MatrixXd X1, int i1, int i2){
  int nr = X1.rows();
  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(nr, nr);
  double tmp;
  
  for(int i = 0; i < nr; i++){
    if(i == (i1 - 1))
      continue;
    tmp = (X1(i1 - 1, i2 - 1) - X1(i, i2 - 1)) ;
    if(tmp > 0){
      s(i1 - 1, i) = s(i, i1 - 1) = (-3. * tmp) / (1. + sqrt(3.) * tmp);
    }else{
      if(tmp == 0){
        s(i1 - 1, i) = s(i, i1 - 1) = 0;
      }else{
        tmp = std::abs(tmp);
        s(i1 - 1, i) = s(i, i1 - 1) = -(-3. * tmp) / (1. + sqrt(3.) * tmp);
      }
    }
  }
  return s;
}

// [[Rcpp::export]]
Eigen::MatrixXd partial_d_dist_abs_dX1_i1_i2_X2_m32(Eigen::MatrixXd X1, Eigen::MatrixXd X2, int i1, int i2){
  int nr = X2.rows();
  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(X1.rows(), nr);
  double tmp;
  
  for(int i = 0; i < nr; i++){
    tmp = X1(i1-1, i2-1) - X2(i, i2-1);
    if(tmp > 0){
      s(i1 - 1, i) = ( - 3. * tmp) / (1. + sqrt(3.) * tmp);
    }else{
      if(tmp == 0){
        s(i1 - 1, i) = 0;
      }else{
        tmp = std::abs(tmp);
        s(i1 - 1, i) = -(-3. * tmp) / (1. + sqrt(3.) * tmp);
      }
    }
  }
  return s;
}



PYBIND11_MODULE(matern, m) {
    m.doc() = "auto-compiled c++ extension";
    m.def("matern5_2_1args",&matern5_2_1args,py::arg("X1"));
    m.def("d_matern5_2_1args_theta_k_iso",&d_matern5_2_1args_theta_k_iso,py::arg("X1"),py::arg("theta"));
    m.def("d_matern5_2_1args_theta_k",&d_matern5_2_1args_theta_k,py::arg("X1"),py::arg("theta"));
    m.def("d_matern5_2_1args_kthetag",&d_matern5_2_1args_kthetag,py::arg("X1"),py::arg("kt"));
    m.def("matern5_2_2args",&matern5_2_2args,py::arg("X1"),py::arg("X2"));
    m.def("d_matern5_2_2args_theta_k_iso",&d_matern5_2_2args_theta_k_iso,py::arg("X1"),py::arg("X2"),py::arg("theta"));
    m.def("d_matern5_2_2args_kthetag",&d_matern5_2_2args_kthetag,py::arg("X1"),py::arg("X2"),py::arg("kt"));
    m.def("partial_d_dist_abs_dX_i1_i2",&partial_d_dist_abs_dX_i1_i2);
    m.def("partial_d_dist_abs_dX1_i1_i2_X2",&partial_d_dist_abs_dX1_i1_i2_X2);
    m.def("matern3_2_1args",&matern3_2_1args,py::arg("X1"));
    m.def("d_matern3_2_1args_theta_k_iso",&d_matern3_2_1args_theta_k_iso,py::arg("X1"),py::arg("theta"));
    m.def("d_matern3_2_1args_theta_k",&d_matern3_2_1args_theta_k,py::arg("X1"),py::arg("theta"));
    m.def("d_matern3_2_1args_kthetag",&d_matern3_2_1args_kthetag,py::arg("X1"),py::arg("kt"));
    m.def("matern3_2_2args",&matern3_2_2args,py::arg("X1"),py::arg("X2"));
    m.def("d_matern3_2_2args_theta_k_iso",&d_matern3_2_2args_theta_k_iso,py::arg("X1"),py::arg("X2"),py::arg("theta"));
    m.def("d_matern3_2_2args_kthetag",&d_matern3_2_2args_kthetag,py::arg("X1"),py::arg("X2"),py::arg("kt"));
}