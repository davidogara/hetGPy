// cppimport
/*<%
cfg['compiler_args'] = ['-std=c++17']
cfg['include_dirs'] = ['../../eigen']
setup_pybind11(cfg)
%>
*/
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <Eigen/LU>


namespace py = pybind11;


// implement pnorm
double pnorm(double q, double mean, double sd, int lower_tail, int log_p){
    double x = (q - mean) / sd; // z transform
    double out = 0.5 * std::erfc(-x * std::sqrt(0.5));
    
    if (lower_tail == 0){
        // return P(X>x)
        out = 1.0 - out;
    }
    if (log_p == 1){
        out = std::log(out);
    }
    return out;
}

// implement dnorm
double dnorm(double x, double mean, double sd, int log_p){
    double c = 1.0 / std::sqrt(2 * M_PI * sd * sd);
    double integrand = -0.5 * pow((x - mean) / sd,2);
    double out = c * std::exp(integrand);
    if (log_p == 1){
        out = std::log(out);
    }
    return out;
}

// [[Rcpp::export]]
double v1cpp(double mu1, double mu2, double s1, double s2, double rho) {
  
  if((std::abs(s1 - s2) < 0.01) & (rho >= 0.99)){
    return(mu1); // Special case when almost perfectly correlated (e.g., points close by)
  }
  double a = sqrt(s1*s1 + s2*s2 - 2*s1*s2*rho);
  double alpha = (mu1 - mu2)/a;
  return(mu1 * pnorm(alpha, 0.0,1.0, 1, 0) + mu2 * pnorm(-alpha,0.0,1.0,1,0) + a * dnorm(alpha,0.0,1.0,0));
}

// [[Rcpp::export]]
double v2cpp(double mu1, double mu2, double s1, double s2, double rho) {
  if((std::abs(s1 - s2) < 0.01) & (rho >= 0.99)){
    return(mu1*mu1 + s1*s1); // Special case when almost perfectly correlated (e.g., points close by)
  }
  double a = sqrt(s1*s1 + s2*s2 - 2*s1*s2*rho);
  double alpha = (mu1 - mu2)/a;
  return(
    (mu1 * mu1 + s1 * s1) * pnorm(alpha, 0.0,1.0, 1, 0) +
      (mu2 * mu2 + s2 * s2) * pnorm(-alpha,0.0,1.0, 1, 0) +
      (mu1 + mu2) * a * dnorm(alpha,0.0,1.0,0));
}

// [[Rcpp::export]]
double r_cpp(double mu1, double mu2, double s1, double s2, double rho,
             double rho1, double rho2) {
  if((std::abs(s1 - s2) < 0.01) & (rho >= 0.99)){
    return(rho1); // Special case when almost perfectly correlated (e.g., points close by)
  }
  
  double a = sqrt(s1*s1 + s2*s2 - 2*s1*s2*rho);
  double alpha = (mu1 - mu2)/a;
  return((
      s1 * rho1 * pnorm(alpha, 0.0,1.0, 1, 0) +
        s2 * rho2 * pnorm(-alpha,0.0,1.0, 1, 0)) /
          sqrt(v2cpp(mu1, mu2, s1, s2, rho) - v1cpp(mu1, mu2, s1, s2, rho)*v1cpp(mu1, mu2, s1, s2, rho)));
}

// [[Rcpp::export]]
double qEI_cpp(Eigen::VectorXd mu, Eigen::VectorXd s, Eigen::MatrixXd cor, double threshold){
  int q = mu.size();
  // if(q < 2){
  //   std::cout << "Error : q < 2" << std::endl;
  // }
  double v1, v2;
  v1 = v1cpp(mu(0), mu(1), s(0), s(1), cor(0,1));
  
  v2 = v2cpp(mu(0), mu(1), s(0), s(1), cor(0,1)) - v1*v1;
  v2 = std::max(v2, 0.);
  
  if(q == 2){
    return(v1cpp(v1, threshold, sqrt(v2), 0.0000001, 0) - threshold);//Difference est la
  }
  
  // The formula works with groups of 3 : max(max(yi-2, yi-1), yi)
  double m1, m2, m3, s1, s2, s3, rho, rho1, rho2, r1, tmp, tmp2;
  m1 = mu(0);
  m2 = mu(1);
  m3 = mu(2);
  s1 = s(0);
  s2 = s(1);
  s3 = s(2);
  rho = cor(0,1);
  rho1 = cor(0,2);
  rho2 = cor(1,2);
  
  for (int i = 2; i < q; i++){
    r1 = r_cpp(m1, m2, s1, s2, rho, rho1, rho2);
    tmp = v1;
    tmp2 = sqrt(v2);
    v1 = v1cpp(tmp, m3, tmp2, s3, r1);
    v2 = v2cpp(tmp, m3, tmp2, s3, r1) - v1*v1;
    v2 = std::max(v2, 0.);
    
    //update
    if(i < q-1){
      rho1 = r_cpp(m1, m2, s1, s2, rho, cor(i-2,i+1), cor(i-1, i+1));
      rho = r1;
      rho2 = cor(i, i+1);
      m1 = tmp;
      m2 = m3;
      m3 = mu(i+1);
      s1 = tmp2;
      s2 = s3;
      s3 = s(i+1);
    }
  }
  
  return(v1cpp(threshold, v1, 0.0000001, sqrt(v2), 0) - threshold);
}


PYBIND11_MODULE(qEI, m) {
    m.doc() = "auto-compiled c++ extension";
    m.def("pnorm",&pnorm,py::arg("q"), py::arg("mean"),py::arg("sd"), py::arg("lower_tail"),py::arg("log_p"));
    m.def("dnorm",&dnorm,py::arg("x"), py::arg("mean"),py::arg("sd"), py::arg("log_p"));
    
    m.def("v1cpp",&v1cpp,py::arg("mu1"), py::arg("mu2"),py::arg("s1"), py::arg("s2"),py::arg("rho"));
    m.def("v2cpp",&v2cpp,py::arg("mu1"), py::arg("mu2"),py::arg("s1"), py::arg("s2"),py::arg("rho"));
    m.def("r_cpp",&r_cpp,py::arg("mu1"), py::arg("mu2"),py::arg("s1"), py::arg("s2"),py::arg("rho"),py::arg("rho1"),py::arg("rho2"));
    m.def("qEI_cpp",&qEI_cpp,py::arg("mu"), py::arg("s"),py::arg("cor"), py::arg("threshold"));
}