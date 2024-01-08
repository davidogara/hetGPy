<%
cfg['compiler_args'] = ['-std=c++17']
cfg['include_dirs'] = ['../eigen']
setup_pybind11(cfg)
%>

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <Eigen/LU>

namespace py = pybind11;


Eigen::MatrixXd d_matern5_2_1args_theta_k(Eigen::MatrixXd X1, double theta){
  // X1 has just one column here
  int nr = X1.rows();
  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(nr,nr);
  double tmp;
  
  const double* ptrX1 = (const double*) &X1(1,0);
  const double* ptrX2 = (const double*) &X1(0,0);
  double* ptrs = &s(0,1);
  double* ptrs2 = &s(1,0); //symmetric
  std::cout << *ptrs << std::endl;;
  for(int i = 1; i < nr; i++, ptrX1++){
    for(int j = 0; j < i; j++, ptrs++){
      tmp = std::abs(*ptrX1 - *ptrX2) / theta;
      
      *ptrs -= ((10./3. - 5.) * tmp - 5. * sqrt(5.)/3. * tmp * tmp) / (1. + sqrt(5.) * tmp + 5./3. * tmp * tmp) * tmp/theta;
      
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


Eigen::MatrixXd d_matern5_2_1args_theta_k_iso(Eigen::MatrixXd X1, double theta){
  int nr = X1.rows();
  int nc = X1.cols();
  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(nr,nr);
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


Eigen::MatrixXd d_matern5_2_1args_kthetag(Eigen::MatrixXd X1, double kt){
  int nr = X1.rows();
  int nc = X1.cols();
  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(nr,nr);
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

PYBIND11_MODULE(matern_utils_eigen, m) {
    m.doc() = "auto-compiled c++ extension";
    m.def("d_matern5_2_1args_theta_k", &d_matern5_2_1args_theta_k);
    m.def("d_matern5_2_1args_theta_k_iso",&d_matern5_2_1args_theta_k_iso);
    m.def("d_matern5_2_1args_kthetag",&d_matern5_2_1args_kthetag);
}