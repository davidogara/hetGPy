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


//////// Gaussian kernel



// [[Rcpp::export]]
Eigen::VectorXd mi_gauss_cpp(Eigen::MatrixXd Mu, Eigen::VectorXd sigma){
  Eigen::VectorXd mis = Eigen::VectorXd::Ones(Mu.rows());

  for(int i = 0; i < Mu.rows(); i++){
    for(int j = 0; j < Mu.cols(); j++){
      mis(i) *= 0.5 * std::sqrt(M_PI) * sigma(j) * (std::erf((1 - Mu(i,j))/sigma(j)) + std::erf(Mu(i,j)/sigma(j)));
    }
  }
  return(mis);
}

// [[Rcpp::export]]
Eigen::MatrixXd Wijs_gauss_cpp(Eigen::MatrixXd Mu1, Eigen::MatrixXd Mu2, Eigen::VectorXd sigma){
  int m1c = Mu1.cols();
  int m2r = Mu2.rows();
  Eigen::MatrixXd Wijs = Eigen::MatrixXd::Ones(Mu1.rows(), m2r);
  double a,b;
  
  for(int i = 0; i < Mu1.rows(); i++){
    for(int j = 0; j < m2r; j++){
      const double* ptr_s = (const double*) &sigma(0);
      for(int k = 0; k < m1c; k++, ptr_s++){
        a = Mu1(i, k);
        b = Mu2(j, k);
        Wijs(i,j) *= -1./2. * std::sqrt(M_PI/2.) * *ptr_s * std::exp(-(a - b) * (a - b) / (2. * *ptr_s * *ptr_s)) * (std::erf((a + b - 2.) / (std::sqrt(2.) * *ptr_s)) - std::erf((a + b) / (std::sqrt(2.) * *ptr_s))) ;
      }
    }
  }
  
  return(Wijs);
}


// [[Rcpp::export]]
Eigen::MatrixXd Wijs_gauss_sym_cpp(Eigen::MatrixXd Mu, Eigen::VectorXd sigma){
  int mc = Mu.cols();
  Eigen::MatrixXd Wijs = Eigen::MatrixXd::Ones(Mu.rows(), Mu.rows());
  double a,b;
  
  for(int i = 0; i < Mu.rows(); i++){
    for(int j = 0; j <= i; j++){
      const double* ptr_s = (const double*) &sigma(0);
      for(int k = 0; k < mc; k++, ptr_s++){
        if(i == j){
          a = Mu(i, k);
          Wijs(i, i) *= -1./2. * std::sqrt(M_PI/2.) * *ptr_s * (std::erf((2. * a  - 2.) / (std::sqrt(2.) * *ptr_s)) - std::erf((2. * a) / (std::sqrt(2.) * *ptr_s)));
        }else{
          a = Mu(i, k);
          b = Mu(j, k);
          Wijs(i,j) = Wijs(j,i) *= -1./2. * std::sqrt(M_PI/2.) * *ptr_s * std::exp(-(a - b) * (a - b) / (2. * *ptr_s * *ptr_s)) * (erf((a + b - 2.) / (std::sqrt(2.) * *ptr_s)) - erf((a + b) / (std::sqrt(2.) * *ptr_s)));
        }
      }
    }
  }
  
  return(Wijs);
}


// Alternative: not recomputing the covariance

// [[Rcpp::export]]
Eigen::VectorXd d_gauss_cpp(Eigen::VectorXd X, double x, double sigma){
  Eigen::VectorXd dis(X.size());  
  
  for(int i = 0; i < X.size(); i++){
    dis(i) = 2. / sigma * (X(i) - x);
  }
  return(dis);
}



double c1i_gauss(double x1, double X, double sigma){
  double tmp = -1./2. * std::sqrt(M_PI/2.) * sigma * std::exp(-(X - x1) * (X - x1) / (2. * sigma * sigma)) * (std::erf((X + x1 - 2.) / (std::sqrt(2.) * sigma)) - std::erf((X + x1) / (std::sqrt(2.) * sigma)));
  if(tmp == 0.) return(0.);
  
  return((0.5*std::exp(-(x1 - X)*(x1 - X) / (2.*sigma*sigma))* (std::exp(-(x1 + X)*(x1 + X)/(2.*sigma*sigma)) -
         std::exp(-(2.-(x1 + X))*(2.-(x1 + X))/(2.*sigma * sigma))) - std::sqrt(2.*M_PI)/4./sigma*(x1 - X) * std::exp(-(x1 - X)*(x1 - X)/(2.*sigma*sigma)) * 
         (std::erf((x1 + X)/(std::sqrt(2.)*sigma)) - std::erf((x1 + X - 2.)/(std::sqrt(2.)*sigma))))/tmp);
}

// [[Rcpp::export]]
double c2_gauss_cpp(double x, double t, double w){
  if(w == 0.) return(0.);
  double tmp = -1./2. * std::sqrt(M_PI/2.) * t * (std::erf((2. * x  - 2.) / (std::sqrt(2.) * t)) - std::erf((2. * x) / (std::sqrt(2.) * t)));
  if(tmp == 0.) return(0.);
  return((std::exp(-2. * x * x / (t * t)) - std::exp(-2.*(1. - x) * (1. - x) / (t * t)))*w/tmp);
}


// [[Rcpp::export]]
Eigen::VectorXd c1_gauss_cpp(Eigen::VectorXd X, double x, double sigma, Eigen::VectorXd W){
  Eigen::VectorXd cis(X.size());  
  
  for(int i = 0; i < X.size(); i++){
    cis(i) = c1i_gauss(x, X(i), sigma) * W(i);
  }
  
  return(cis);
}

//////// Matern 5/2 kernel

double A_2_cpp(double x){
  return((8. + 5. * x + x * x) * std::exp(-x));
}

// [[Rcpp::export]]
Eigen::VectorXd mi_mat52_cpp(Eigen::MatrixXd Mu, Eigen::VectorXd sigma){
  Eigen::VectorXd mis = Eigen::VectorXd::Ones(Mu.rows());

  for(int i = 0; i < Mu.rows(); i++){
    for(int j = 0; j < Mu.cols(); j++){
      mis(i) *= sigma(j)/(3.*std::sqrt(5.)) * (16. - A_2_cpp(std::sqrt(5.) * Mu(i,j)/sigma(j)) - A_2_cpp(std::sqrt(5.) * (1. - Mu(i,j))/sigma(j)));
    }
  }
  return(mis);
}



// [[Rcpp::export]]
Eigen::MatrixXd Wijs_mat52_cpp(Eigen::MatrixXd Mu1, Eigen::MatrixXd Mu2, Eigen::VectorXd sigma){
  
  int m1c = Mu1.cols();
  int m2r = Mu2.rows();
  double tmp;
  Eigen::MatrixXd Wijs = Eigen::MatrixXd::Ones(Mu1.rows(), m2r);
  
  double a,a2,b,b2,t,t2;
  double p1, p3, p4;
  
  for(int i = 0; i < Mu1.rows(); i++){
    for(int j = 0; j < m2r; j++){
      const double* ptr_s = (const double*) &sigma(0);
      for(int k = 0; k < m1c; k++, ptr_s++){
        a = Mu1(i,k);
        b = Mu2(j,k);
        
        if(b < a){
          tmp = b; 
          b = a;
          a = tmp;
        }
        
        t = *ptr_s;
        t2 = t*t;
        a2 = a*a;
        b2 = b*b;
        p1 = (2*t2*(63*t2 + 9*5.*std::sqrt(5.)*b*t-9*5.*std::sqrt(5.)*a*t+50*b2-100*a*b+50*a2)*std::exp(2*std::sqrt(5.)*a/t)-63*t2*t2-9*5.*std::sqrt(5.)*(b+a)*t*t2-10*(5*b2+17*a*b+5*a2)*t2-8*5.*std::sqrt(5.)*a*b*(b+a)*t-50*a2*b2)*std::exp(-std::sqrt(5.)*(b+a)/t)/(36*std::sqrt(5.)*t*t2);
        p3 = (b-a)*(54*t2*t2+(54*std::sqrt(5.)*b-54*std::sqrt(5.)*a)*t*t2+(105*b2-210*a*b+105*a2)*t2+(3*5.*std::sqrt(5.)*b2*b-9*5.*std::sqrt(5.)*a*b2+9*5.*std::sqrt(5.)*a2*b-3*5.*std::sqrt(5.)*a2*a)*t+5*b2*b2-20*a*b2*b+30*a2*b2-20*a2*a*b+5*a2*a2)*std::exp(std::sqrt(5.)*(a-b)/t)/(54*t2*t2);
        p4 = -((t*(t*(9*t*(7*t-5.*std::sqrt(5.)*(b+a-2))+10*b*(5*b+17*a-27)+10*(5*a2-27*a+27))-8*5.*std::sqrt(5.)*(a-1)*(b-1)*(b+a-2))+50*(a-1)*(a-1)*(b-2)*b+50*(a-1)*(a-1))*std::exp(2*std::sqrt(5.)*b/t))*std::exp(-std::sqrt(5.)*(b-a+2)/t)/(36*std::sqrt(5.)*t*t2);
        
        Wijs(i,j) *= p1 + p3 + p4;
        
      }
    }
  } 
  
  return(Wijs);   
}



// [[Rcpp::export]]
Eigen::MatrixXd Wijs_mat52_sym_cpp(Eigen::MatrixXd Mu, Eigen::VectorXd sigma){
  int m1c = Mu.cols();
  Eigen::MatrixXd Wijs = Eigen::MatrixXd::Ones(Mu.rows(), Mu.rows());
  
  double a,a2,b,b2,t,t2;
  double p1, p3, p4;
  double tmp;
  
  for(int i = 0; i < Mu.rows(); i++){
    for(int j = 0; j <= i; j++){
      const double* ptr_s = (const double*) &sigma(0);
      for(int k = 0; k < m1c; k++, ptr_s++){
        a = Mu(j,k);
        b = Mu(i,k);
        if(b < a){
          tmp = b;
          b = a;
          a = tmp;
        }
        
        t = *ptr_s;
        t2 = t*t;
        a2 = a*a;
        b2 = b*b;
        
        if(i == j){
          Wijs(i,j) *= (std::exp(-2*std::sqrt(5.)*a/t)*(63*t2*t2*std::exp(2*std::sqrt(5.)*a/t)-50*a2*a2-16*5*std::sqrt(5.)*t*a2*a-270*t2*a2-18*5*std::sqrt(5.)*t2*t*a-63*t2*t2)-std::exp(-2*std::sqrt(5.)/t)*((t*(t*(10*(5*a2-27*a+27)+9*t*(7*t-5*std::sqrt(5.)*(2*a-2))+10*a*(22*a-27))-8*5*std::sqrt(5.)*(a-1)*(a-1)*(2*a-2))+50*(a-2)*(a-1)*(a-1)*a+50*(a-1)*(a-1))*std::exp(2*std::sqrt(5.)*a/t)-63*t2*t2*std::exp(2*std::sqrt(5.)/t)))/(36*std::sqrt(5.)*t2*t);;
        }else{
          p1 = (2*t2*(63*t2 + 9*5.*std::sqrt(5.)*b*t-9*5.*std::sqrt(5.)*a*t+50*b2-100*a*b+50*a2)*std::exp(2*std::sqrt(5.)*a/t)-63*t2*t2-9*5.*std::sqrt(5.)*(b+a)*t*t2-10*(5*b2+17*a*b+5*a2)*t2-8*5.*std::sqrt(5.)*a*b*(b+a)*t-50*a2*b2)*std::exp(-std::sqrt(5.)*(b+a)/t)/(36*std::sqrt(5.)*t*t2);
          p3 = (b-a)*(54*t2*t2+(54*std::sqrt(5.)*b-54*std::sqrt(5.)*a)*t*t2+(105*b2-210*a*b+105*a2)*t2+(3*5.*std::sqrt(5.)*b2*b-9*5.*std::sqrt(5.)*a*b2+9*5.*std::sqrt(5.)*a2*b-3*5.*std::sqrt(5.)*a2*a)*t+5*b2*b2-20*a*b2*b+30*a2*b2-20*a2*a*b+5*a2*a2)*std::exp(std::sqrt(5.)*(a-b)/t)/(54*t2*t2);
          p4 = -((t*(t*(9*t*(7*t-5.*std::sqrt(5.)*(b+a-2))+10*b*(5*b+17*a-27)+10*(5*a2-27*a+27))-8*5.*std::sqrt(5.)*(a-1)*(b-1)*(b+a-2))+50*(a-1)*(a-1)*(b-2)*b+50*(a-1)*(a-1))*std::exp(2*std::sqrt(5.)*b/t))*std::exp(-std::sqrt(5.)*(b-a+2)/t)/(36*std::sqrt(5.)*t*t2);
          
          Wijs(j,i) = Wijs(i,j) *= p1 + p3 + p4;
        }
        
      }
    }
  }
  
  return(Wijs);   
}


double c1i_mat52(double a, double b, double t){
  double p1,p3,p4,t2,a2,b2,dw;
  
  bool boo = true;
  
  if(b < a){
    double c = b;
    b = a;
    a = c;
    boo = false;
  }
  
  
  t2 = t*t;
  a2 = a*a;
  b2 = b*b;
  
  p1 = (2*t2*(63*t2 + 9*5.*std::sqrt(5.)*b*t-9*5.*std::sqrt(5.)*a*t+50*b2-100*a*b+50*a2)*std::exp(2*std::sqrt(5.)*a/t)-63*t2*t2-9*5.*std::sqrt(5.)*(b+a)*t*t2-10*(5*b2+17*a*b+5*a2)*t2-8*5.*std::sqrt(5.)*a*b*(b+a)*t-50*a2*b2)*std::exp(-std::sqrt(5.)*(b+a)/t)/(36*std::sqrt(5.)*t*t2);
  p3 = (b-a)*(54*t2*t2+(54*std::sqrt(5.)*b-54*std::sqrt(5.)*a)*t*t2+(105*b2-210*a*b+105*a2)*t2+(3*5.*std::sqrt(5.)*b2*b-9*5.*std::sqrt(5.)*a*b2+9*5.*std::sqrt(5.)*a2*b-3*5.*std::sqrt(5.)*a2*a)*t+5*b2*b2-20*a*b2*b+30*a2*b2-20*a2*a*b+5*a2*a2)*std::exp(std::sqrt(5.)*(a-b)/t)/(54*t2*t2);
  p4 = -((t*(t*(9*t*(7*t-5.*std::sqrt(5.)*(b+a-2))+10*b*(5*b+17*a-27)+10*(5*a2-27*a+27))-8*5.*std::sqrt(5.)*(a-1)*(b-1)*(b+a-2))+50*(a-1)*(a-1)*(b-2)*b+50*(a-1)*(a-1))*std::exp(2*std::sqrt(5.)*b/t))*std::exp(-std::sqrt(5.)*(b-a+2)/t)/(36*std::sqrt(5.)*t*t2);
  
  if((p1 + p3 + p4) == 0.) return(0.); 
  
  if(!boo){
    dw = -((2*5*std::sqrt(5.)*std::exp(2*std::sqrt(5.)/t)*a2*a2*a+(-100*t-2*5*5*std::sqrt(5.)*b)*std::exp(2*std::sqrt(5.)/t)*a2*a2+(18*5*std::sqrt(5.)*t2+400*b*t+4*5*5*std::sqrt(5.)*b2)*std::exp(2*std::sqrt(5.)/t)*a2*a+((150*t2*t+(24*5*std::sqrt(5.)-24*5*std::sqrt(5.)*b)*t2+(150*b2-300*b+150)*t)*std::exp(2*std::sqrt(5.)*b/t)+
      (-210*t2*t-54*5*std::sqrt(5.)*b*t2-600*b2*t-4*5*5*std::sqrt(5.)*b2*b)*std::exp(2*std::sqrt(5.)/t))*a2+((-3*5*5*std::sqrt(5.)*t2*t2+(270*b-570)*t2*t+(-12*5*std::sqrt(5.)*b2+72*5*std::sqrt(5.)*b-12*5*5*std::sqrt(5.))*t2+(-300*b2+600*b-300)*t)*std::exp(2*std::sqrt(5.)*b/t)+(42*std::sqrt(5.)*t2*t2+420*b*t2*t+
      54*5*std::sqrt(5.)*b2*t2+400*b2*b*t+2*5*5*std::sqrt(5.)*b2*b2)*std::exp(2*std::sqrt(5.)/t))*a+(54*t2*t2*t+(108*std::sqrt(5.)-33*std::sqrt(5.)*b)*t2*t2+(30*b2-330*b+450)*t2*t+(12*5*std::sqrt(5.)*b2-48*5*std::sqrt(5.)*b+36*5*std::sqrt(5.))*t2+(150*b2-300*b+150)*t)*std::exp(2*std::sqrt(5.)*b/t)+(-42*std::sqrt(5.)*b*t2*t2-
      210*b2*t2*t-18*5*std::sqrt(5.)*b2*b*t2-100*b2*b2*t-2*5*std::sqrt(5.)*b2*b2*b)*std::exp(2*std::sqrt(5.)/t))*std::exp(2*std::sqrt(5.)*a/t)+(-150*t2*t-24*5*std::sqrt(5.)*b*t2-150*b2*t)*std::exp(2*std::sqrt(5.)/t)*a2+(-3*5*5*std::sqrt(5.)*t2*t2-270*b*t2*t-12*5*std::sqrt(5.)*b2*t2)*std::exp(2*std::sqrt(5.)/t)*a+(-54*t2*t2*t-33*std::sqrt(5.)*b*t2*t2-
      30*b2*t2*t)*std::exp(2*std::sqrt(5.)/t))*std::exp(-std::sqrt(5.)*a/t-std::sqrt(5.)*b/t-2*std::sqrt(5.)/t)/(108*t2*t2*t);
  }else{
    dw = -(((150*t2*t+(24*5*std::sqrt(5.)-24*5*std::sqrt(5.)*a)*t2+(150*a2-300*a+150)*t)*std::exp(2*std::sqrt(5.)*a/t)*b2+(-3*5*5*std::sqrt(5.)*t2*t2+(270*a-570)*t2*t+(-12*5*std::sqrt(5.)*a2+72*5*std::sqrt(5.)*a-12*5*5*std::sqrt(5.))*t2+(-300*a2+600*a-300)*t)*std::exp(2*std::sqrt(5.)*a/t)*b+
      (54*t2*t2*t+(108*std::sqrt(5.)-33*std::sqrt(5.)*a)*t2*t2+(30*a2-330*a+450)*t2*t+(12*5*std::sqrt(5.)*a2-48*5*std::sqrt(5.)*a+36*5*std::sqrt(5.))*t2+(150*a2-300*a+150)*t)*std::exp(2*std::sqrt(5.)*a/t))*std::exp(2*std::sqrt(5.)*b/t)+2*5*std::sqrt(5.)*std::exp(2*std::sqrt(5.)*a/t+2*std::sqrt(5.)/t)*b2*b2*b+
      (100*t-2*5*5*std::sqrt(5.)*a)*std::exp(2*std::sqrt(5.)*a/t+2*std::sqrt(5.)/t)*b2*b2+(18*5*std::sqrt(5.)*t2-400*a*t+4*5*5*std::sqrt(5.)*a2)*std::exp(2*std::sqrt(5.)*a/t+2*std::sqrt(5.)/t)*b2*b+((210*t2*t-54*5*std::sqrt(5.)*a*t2+600*a2*t-4*5*5*std::sqrt(5.)*a2*a)*std::exp(2*std::sqrt(5.)*a/t+
      2*std::sqrt(5.)/t)+(-150*t2*t-24*5*std::sqrt(5.)*a*t2-150*a2*t)*std::exp(2*std::sqrt(5.)/t))*b2+((42*std::sqrt(5.)*t2*t2-420*a*t2*t+54*5*std::sqrt(5.)*a2*t2-400*a2*a*t+2*5*5*std::sqrt(5.)*a2*a2)*std::exp(2*std::sqrt(5.)*a/t+2*std::sqrt(5.)/t)+
      (-3*5*5*std::sqrt(5.)*t2*t2-270*a*t2*t-12*5*std::sqrt(5.)*a2*t2)*std::exp(2*std::sqrt(5.)/t))*b+(-42*std::sqrt(5.)*a*t2*t2+210*a2*t2*t-18*5*std::sqrt(5.)*a2*a*t2+100*a2*a2*t-2*5*std::sqrt(5.)*a2*a2*a)*std::exp(2*std::sqrt(5.)*a/t+2*std::sqrt(5.)/t)+
      (-54*t2*t2*t-33*std::sqrt(5.)*a*t2*t2-30*a2*t2*t)*std::exp(2*std::sqrt(5.)/t))*std::exp(-std::sqrt(5.)*b/t-std::sqrt(5.)*a/t-2*std::sqrt(5.)/t)/(108*t2*t2*t);
  }
  
  return(dw/(p1+p3+p4));
}

// [[Rcpp::export]]
double c2_mat52_cpp(double x, double t, double w){
  double x2 = x*x;
  double t2 = t*t;
  if(w == 0.) return(0.);
  
  double tmp = (std::exp(-2*std::sqrt(5.)*x/t)*(63*t2*t2*std::exp(2*std::sqrt(5.)*x/t)-50*x2*x2-16*5*std::sqrt(5.)*t*x2*x-270*t2*x2-18*5*std::sqrt(5.)*t2*t*x-63*t2*t2)-std::exp(-2*std::sqrt(5.)/t)*((t*(t*(10*(5*x2-27*x+27)+9*t*(7*t-5*std::sqrt(5.)*(2*x-2))+10*x*(22*x-27))-8*5*std::sqrt(5.)*(x-1)*(x-1)*(2*x-2))+50*(x-2)*(x-1)*(x-1)*x+50*(x-1)*(x-1))*std::exp(2*std::sqrt(5.)*x/t)-63*t2*t2*std::exp(2*std::sqrt(5.)/t)))/(36*std::sqrt(5.)*t2*t);
  if(tmp == 0.) return(0.);
  double dw = -((25*x2*x2-2*(3*5*std::sqrt(5.)*t+50)*x2*x+3*(t*(25*t+6*5*std::sqrt(5.))+50)*x2-2*(3*t*(t*(3*std::sqrt(5.)*t+25)+3*5*std::sqrt(5.))+50)*x+9*t2*t2+18*std::sqrt(5.)*t2*t+75*t2+6*5*std::sqrt(5.)*t+25)*std::exp(4*std::sqrt(5.)*x/t)-25*std::exp(2*std::sqrt(5.)/t)*x2*x2-6*5*std::sqrt(5.)*t*std::exp(2*std::sqrt(5.)/t)*x2*x-75*t2*std::exp(2*std::sqrt(5.)/t)*x2-18*std::sqrt(5.)*t2*t*std::exp(2*std::sqrt(5.)/t)*x-9*t2*t2*std::exp(2*std::sqrt(5.)/t))*std::exp(-2*std::sqrt(5.)*(x+1)/t)/(9*t2*t2);
  return(dw*w/tmp);
}


// [[Rcpp::export]]
Eigen::VectorXd c1_mat52_cpp(Eigen::VectorXd X, double x, double sigma, Eigen::VectorXd W){
  Eigen::VectorXd cis(X.size());  
  
  for(int i = 0; i < X.size(); i++){
    cis(i) += c1i_mat52(X(i), x, sigma) * W(i);
  }
  
  return(cis);
}

// [[Rcpp::export]]
Eigen::VectorXd d_mat52_cpp(Eigen::VectorXd X, double x, double sigma){
  Eigen::VectorXd s(X.size());
  double tmp;
  
  for(int i = 0; i < X.size(); i++){
    tmp = (x - X(i))/sigma;
    if(tmp > 0){
      s(i) = ((10./3. - 5.) * tmp - 5 * std::sqrt(5.)/3. * tmp * tmp) / (1. + std::sqrt(5.) * tmp + 5./3. * tmp * tmp);
    }else{
      if(tmp == 0){
        s(i) = 0;
      }else{
        tmp = std::abs(tmp);
        s(i) = -((10./3. - 5.) * tmp - 5. * std::sqrt(5.)/3. * tmp * tmp) / ((1. + std::sqrt(5.) * tmp + 5./3. * tmp * tmp));
      }
    }
  }
  return(s/sigma);
}

//////// Matern 3/2 kernel

double A_1_cpp(double x){
  return((2. + x) * std::exp(-x));
}

// [[Rcpp::export]]
Eigen::VectorXd mi_mat32_cpp(Eigen::MatrixXd Mu, Eigen::VectorXd sigma){
  Eigen::VectorXd mis = Eigen::VectorXd::Ones(Mu.rows());
  
  for(int i = 0; i < Mu.rows(); i++){
    for(int j = 0; j < Mu.cols(); j++){
      mis(i) *= sigma(j)/(std::sqrt(3.)) * (4. - A_1_cpp(std::sqrt(3.) * Mu(i,j)/sigma(j)) - A_1_cpp(std::sqrt(3.) * (1. - Mu(i,j))/sigma(j)));
    }
  }
  return(mis);
}

// [[Rcpp::export]]
Eigen::MatrixXd Wijs_mat32_cpp(Eigen::MatrixXd Mu1, Eigen::MatrixXd Mu2, Eigen::VectorXd sigma){
  
  int m1c = Mu1.cols();
  int m2r = Mu2.rows();
  double tmp;
  Eigen::MatrixXd Wijs = Eigen::MatrixXd::Ones(Mu1.rows(), m2r);
  
  double a,b,t,t2;
  double p;
  
  for(int i = 0; i < Mu1.rows(); i++){
    for(int j = 0; j < m2r; j++){
      const double* ptr_s = (const double*) &sigma(0);
      for(int k = 0; k < m1c; k++, ptr_s++){
        a = Mu1(i,k);
        b = Mu2(j,k);
        
        if(b < a){
          tmp = b; 
          b = a;
          a = tmp;
        }
        
        t = *ptr_s;
        t2 = t*t;
        p = ((t*(5*std::sqrt(3.)*t+9*b-9*a)*std::exp((2*std::sqrt(3.)*a)/t)-5*std::sqrt(3.)*t2-9*(b+a)*t-2*3*std::sqrt(3.)*a*b)*std::exp(-(std::sqrt(3.)*(b+a))/t))/(12*t)+((b-a)*(2*t2+2*std::sqrt(3.)*(b-a)*t+b*b-2*a*b+a*a)*std::exp(-(std::sqrt(3.)*(b-a))/t))/(2*t2)-(((t*(5*t-3*std::sqrt(3.)*(b+a-2))+6*(a-1)*b-6*a+6)*std::exp((2*std::sqrt(3.)*b)/t)-t*(5*t+3*std::sqrt(3.)*(b-a))*std::exp((2*std::sqrt(3.))/t))*std::exp(-(std::sqrt(3.)*(b-a+2))/t))/(4*std::sqrt(3.)*t);
        
        Wijs(i,j) *= p;
        
      }
    }
  } 
  
  return(Wijs);   
}

// [[Rcpp::export]]
Eigen::MatrixXd Wijs_mat32_sym_cpp(Eigen::MatrixXd Mu, Eigen::VectorXd sigma){
  int m1c = Mu.cols();
  Eigen::MatrixXd Wijs = Eigen::MatrixXd::Ones(Mu.rows(), Mu.rows());
  
  double a,b,t,t2;
  double p;
  double tmp;
  
  for(int i = 0; i < Mu.rows(); i++){
    for(int j = 0; j <= i; j++){
      const double* ptr_s = (const double*) &sigma(0);
      for(int k = 0; k < m1c; k++, ptr_s++){
        a = Mu(j,k);
        b = Mu(i,k);
        if(b < a){
          tmp = b;
          b = a;
          a = tmp;
        }
        
        t = *ptr_s;
        t2 = t*t;
        
        if(i == j){
          Wijs(i,j) *= (15*t2-(t*(15*t-2*9*std::sqrt(3.)*(a-1))+18*(a-1)*(a-1))*std::exp((2*std::sqrt(3.)*a)/t-(2*std::sqrt(3.))/t))/(4*3*std::sqrt(3.)*t) -((5*t2+2*3*std::sqrt(3.)*a*t+6*a*a)*std::exp(-(2*std::sqrt(3.)*a)/t)-5*t2)/(4*std::sqrt(3.)*t);
        }else{
          p = ((t*(5*std::sqrt(3.)*t+9*b-9*a)*std::exp((2*std::sqrt(3.)*a)/t)-5*std::sqrt(3.)*t2-9*(b+a)*t-2*3*std::sqrt(3.)*a*b)*std::exp(-(std::sqrt(3.)*(b+a))/t))/(12*t)+((b-a)*(2*t2+2*std::sqrt(3.)*(b-a)*t+b*b-2*a*b+a*a)*std::exp(-(std::sqrt(3.)*(b-a))/t))/(2*t2)-(((t*(5*t-3*std::sqrt(3.)*(b+a-2))+6*(a-1)*b-6*a+6)*std::exp((2*std::sqrt(3.)*b)/t)-t*(5*t+3*std::sqrt(3.)*(b-a))*std::exp((2*std::sqrt(3.))/t))*std::exp(-(std::sqrt(3.)*(b-a+2))/t))/(4*std::sqrt(3.)*t);
          
          Wijs(j,i) = Wijs(i,j) *= p;
        }
        
      }
    }
  }
  
  return(Wijs);   
}

// [[Rcpp::export]]
Eigen::VectorXd d_mat32_cpp(Eigen::VectorXd X, double x, double sigma){
  Eigen::VectorXd s(X.size());
  double tmp;
  
  for(int i = 0; i < X.size(); i++){
    tmp = (x - X(i))/sigma;
    if(tmp > 0){
      s(i) = -3*tmp / (1 + std::sqrt(3.) * tmp);
    }else{
      if(tmp == 0){
        s(i) = 0;
      }else{
        tmp = -tmp;
        s(i) = 3*tmp / (1 + std::sqrt(3.) * tmp);
      }
    }
  }
  return(s/sigma);
}

double c1i_mat32(double a, double b, double t){
  double a2, b2, p,t2,dw;
  
  bool boo = true;
  
  if(b < a){
    double c = b;
    b = a;
    a = c;
    boo = false;
  }
  
  a2 = a*a;
  b2 = b*b;
  
  t2 = t*t;
  
  p = ((t*(5*std::sqrt(3.)*t+9*b-9*a)*std::exp((2*std::sqrt(3.)*a)/t)-5*std::sqrt(3.)*t2-9*(b+a)*t-2*3*std::sqrt(3.)*a*b)*std::exp(-(std::sqrt(3.)*(b+a))/t))/(12*t)+((b-a)*(2*t2+2*std::sqrt(3.)*(b-a)*t+b*b-2*a*b+a*a)*std::exp(-(std::sqrt(3.)*(b-a))/t))/(2*t2)-(((t*(5*t-3*std::sqrt(3.)*(b+a-2))+6*(a-1)*b-6*a+6)*std::exp((2*std::sqrt(3.)*b)/t)-t*(5*t+3*std::sqrt(3.)*(b-a))*std::exp((2*std::sqrt(3.))/t))*std::exp(-(std::sqrt(3.)*(b-a+2))/t))/(4*std::sqrt(3.)*t);

  if(p == 0.) return(0.);
  
  if(!boo){
    dw =-(((2*std::sqrt(3.)*std::exp((2*std::sqrt(3.))/t)*a2*a+(-6*t-2*3*std::sqrt(3.)*b)*std::exp((2*std::sqrt(3.))/t)*a2+(((6*b-6)*t-3*std::sqrt(3.)*t2)*std::exp((2*std::sqrt(3.)*b)/t)+(2*std::sqrt(3.)*t2+12*b*t+2*3*std::sqrt(3.)*b2)*std::exp((2*std::sqrt(3.))/t))*a+(2*t2*t+(4*std::sqrt(3.)-std::sqrt(3.)*b)*t2+(6-6*b)*t)*std::exp((2*std::sqrt(3.)*b)/t)+(-2*std::sqrt(3.)*b*t2-6*b2*t-2*std::sqrt(3.)*b2*b)*std::exp((2*std::sqrt(3.))/t))*std::exp((2*std::sqrt(3.)*a)/t)+(-3*std::sqrt(3.)*t2-6*b*t)*std::exp((2*std::sqrt(3.))/t)*a+(-2*t2*t-std::sqrt(3.)*b*t2)*std::exp((2*std::sqrt(3.))/t))*std::exp((-std::sqrt(3.)*a-std::sqrt(3.)*b-2*std::sqrt(3.))/t))/(4*t2*t);
  }else{
    dw = ((t*((3*std::sqrt(3.)*t-6*a+6)*b-t*(2*t-std::sqrt(3.)*(a-4))+6*a-6)*std::exp((2*std::sqrt(3.)*(b+a))/t)-2*std::sqrt(3.)*std::exp((2*std::sqrt(3.)*(a+1))/t)*b2*b-2*(3*t-3*std::sqrt(3.)*a)*std::exp((2*std::sqrt(3.)*(a+1))/t)*b2-std::exp((2*std::sqrt(3.))/t)*(2*std::sqrt(3.)*t2*std::exp((2*std::sqrt(3.)*a)/t)-12*a*t*std::exp((2*std::sqrt(3.)*a)/t)+2*3*std::sqrt(3.)*a2*std::exp((2*std::sqrt(3.)*a)/t)-3*std::sqrt(3.)*t2-6*a*t)*b+2*a*(std::sqrt(3.)*t2-3*a*t+std::sqrt(3.)*a2)*std::exp((2*std::sqrt(3.)*(a+1))/t)+t2*(2*t+std::sqrt(3.)*a)*std::exp((2*std::sqrt(3.))/t))*std::exp(-(std::sqrt(3.)*(b+a+2))/t))/(4*t2*t);
  }
  
  return(dw/p);
}

// [[Rcpp::export]]
double c2_mat32_cpp(double x, double t, double w){
  double x2 = x*x;
  double t2 = t*t;
  
  if(w == 0.) return(0.);
  
  double tmp = (15*t2-(t*(15*t-2*9*std::sqrt(3.)*(x-1))+18*(x-1)*(x-1))*std::exp((2*std::sqrt(3.)*x)/t-(2*std::sqrt(3.))/t))/(4*3*std::sqrt(3.)*t) -((5*t2+2*3*std::sqrt(3.)*x*t+6*x*x)*std::exp(-(2*std::sqrt(3.)*x)/t)-5*t2)/(4*std::sqrt(3.)*t);
  
  if(tmp == 0.) return(0.);
  
  double dw = -(((3*x*x-2*(std::sqrt(3.)*t+3)*x+t2+2*std::sqrt(3.)*t+3)*std::exp((4*std::sqrt(3.)*x)/t)-3*std::exp((2*std::sqrt(3.))/t)*x2-2*std::sqrt(3.)*t*std::exp((2*std::sqrt(3.))/t)*x-t2*std::exp((2*std::sqrt(3.))/t))*std::exp(-(2*std::sqrt(3.)*(x+1))/t))/t2;
  return(dw*w/tmp);
}

// [[Rcpp::export]]
Eigen::VectorXd c1_mat32_cpp(Eigen::VectorXd X, double x, double sigma, Eigen::VectorXd W){
  Eigen::VectorXd cis(X.size());
  
  for(int i = 0; i < X.size(); i++){
    cis(i) += c1i_mat32(X(i), x, sigma) * W(i);
  }
  
  return(cis);
}


PYBIND11_MODULE(EMSE, m) {
    m.doc() = "auto-compiled c++ extension";
    m.def("mi_gauss_cpp",&mi_gauss_cpp,py::arg("Mu"), py::arg("sigma"));
    m.def("Wijs_gauss_cpp",&Wijs_gauss_cpp,py::arg("Mu1"), py::arg("Mu2"), py::arg("sigma"));
    m.def("Wijs_gauss_sym_cpp",&Wijs_gauss_sym_cpp,py::arg("Mu"), py::arg("sigma"));
    m.def("d_gauss_cpp",&d_gauss_cpp,py::arg("X"), py::arg("x"), py::arg("sigma"));
    m.def("c2_gauss_cpp",&c2_gauss_cpp,py::arg("x"), py::arg("t"), py::arg("w"));
    m.def("c1_gauss_cpp",&c1_gauss_cpp,py::arg("X"), py::arg("x"), py::arg("sigma"), py::arg("W"));
    m.def("mi_mat52_cpp",&mi_mat52_cpp,py::arg("Mu"), py::arg("sigma"));
    m.def("Wijs_mat52_cpp",&Wijs_mat52_cpp,py::arg("Mu1"), py::arg("Mu2"), py::arg("sigma"));
    m.def("Wijs_mat52_sym_cpp",&Wijs_mat52_sym_cpp,py::arg("Mu"), py::arg("sigma"));
    m.def("c2_mat52_cpp",&c2_mat52_cpp,py::arg("x"), py::arg("t"), py::arg("w"));
    m.def("c1_mat52_cpp",&c1_mat52_cpp,py::arg("X"), py::arg("x"), py::arg("sigma"), py::arg("W"));
    m.def("d_mat52_cpp",&d_mat52_cpp,py::arg("X"), py::arg("x"), py::arg("sigma"));
    m.def("mi_mat32_cpp",&mi_mat32_cpp,py::arg("Mu"), py::arg("sigma"));
    m.def("Wijs_mat32_cpp",&Wijs_mat32_cpp,py::arg("Mu1"), py::arg("Mu2"), py::arg("sigma"));
    m.def("Wijs_mat32_sym_cpp",&Wijs_mat32_sym_cpp,py::arg("Mu"), py::arg("sigma"));
    m.def("d_mat32_cpp",&d_mat32_cpp,py::arg("X"), py::arg("x"), py::arg("sigma"));
    m.def("c2_mat32_cpp",&c2_mat32_cpp,py::arg("x"), py::arg("t"), py::arg("w"));
    m.def("c1_mat32_cpp",&c1_mat32_cpp,py::arg("X"), py::arg("x"), py::arg("sigma"), py::arg("W"));
}
