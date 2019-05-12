function [Z_bn,Z_hat,gamma, Z_mu, ivar, var_sqrt, var, eps] = batch_norm_forward_Test(Z,gamma,beta,mu,sigma)
eps = 10^-5;

Z_mu = Z - mu;
var = sigma.^2;
var_sqrt = (sqrt(var + eps));
ivar = 1./var_sqrt;
Z_hat = Z_mu .* ivar;
Z_gamma = gamma .* Z_hat;
Z_bn = Z_gamma + beta;
