// cppimport
//<%
//cfg['compiler_args'] = ['-std=c++17']
//cfg['include_dirs'] = ['../eigen']
//setup_pybind11(cfg)
//%>

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <Eigen/LU>

namespace py = pybind11;

// compile partial derivatives for Gaussian covariance

// [[Rcpp::export]]
Eigen::MatrixXd partial_d_dist_dX_i1_i2(Eigen::MatrixXd X1, int i1, int i2){
  int nr = X1.rows();
  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(nr, nr);

  for(int i = 0; i < nr; i++){
    if(i == (i1 - 1))
      continue;
    s(i1 - 1, i) = s(i, i1 - 1) = -2 * (X1(i1-1, i2-1) - X1(i, i2-1));
  }
  return s;
}

// [[Rcpp::export]]
Eigen::MatrixXd partial_d_dist_dX1_i1_i2_X2(Eigen::MatrixXd X1, Eigen::MatrixXd X2, int i1, int i2){
  int nr = X2.rows();
  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(X1.rows(), nr);

  for(int i = 0; i < nr; i++){
    s(i1 - 1, i) = -2 * (X1(i1-1, i2-1) - X2(i, i2-1));
  }
  return s;
}

PYBIND11_MODULE(gauss, m) {
    m.doc() = "auto-compiled c++ extension";
    m.def("partial_d_dist_dX_i1_i2",&partial_d_dist_dX_i1_i2,py::arg("X1"),py::arg("i1"),py::arg("i2"));
    m.def("partial_d_dist_dX1_i1_i2_X2",&partial_d_dist_dX1_i1_i2_X2,py::arg("X1"),py::arg("X2"),py::arg("i1"),py::arg("i2"));
}

