#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double v1cpp(double mu1, double mu2, double s1, double s2, double rho) {
  
  if((std::abs(s1 - s2) < 0.01) & (rho >= 0.99)){
    return(mu1); // Special case when almost perfectly correlated (e.g., points close by)
  }
  double a = sqrt(s1*s1 + s2*s2 - 2*s1*s2*rho);
  double alpha = (mu1 - mu2)/a;
  return(mu1 * R::pnorm(alpha, 0.0,1.0, 1, 0) + mu2 * R::pnorm(-alpha,0.0,1.0,1,0) + a * R::dnorm(alpha,0.0,1.0,0));
}

// [[Rcpp::export]]
double v2cpp(double mu1, double mu2, double s1, double s2, double rho) {
  if((std::abs(s1 - s2) < 0.01) & (rho >= 0.99)){
    return(mu1*mu1 + s1*s1); // Special case when almost perfectly correlated (e.g., points close by)
  }
  double a = sqrt(s1*s1 + s2*s2 - 2*s1*s2*rho);
  double alpha = (mu1 - mu2)/a;
  return(
    (mu1 * mu1 + s1 * s1) * R::pnorm(alpha, 0.0,1.0, 1, 0) +
      (mu2 * mu2 + s2 * s2) * R::pnorm(-alpha,0.0,1.0, 1, 0) +
      (mu1 + mu2) * a * R::dnorm(alpha,0.0,1.0,0));
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
      s1 * rho1 * R::pnorm(alpha, 0.0,1.0, 1, 0) +
        s2 * rho2 * R::pnorm(-alpha,0.0,1.0, 1, 0)) /
          sqrt(v2cpp(mu1, mu2, s1, s2, rho) - v1cpp(mu1, mu2, s1, s2, rho)*v1cpp(mu1, mu2, s1, s2, rho)));
}

// [[Rcpp::export]]
double qEI_cpp(NumericVector mu, NumericVector s, NumericMatrix cor, double threshold){
  int q = mu.length();
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


