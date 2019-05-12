
function[dZ,d_gamma,d_beta] = batch_norm_backward(dZ_bn,Z_hat,gamma, Z_mu, ivar, var_sqrt, var, eps)
num_training_samples = size(dZ_bn,2);
num_features = size(dZ_bn,1);

d_beta = sum(dZ_bn,2);
assert(size(d_beta,1) == size(dZ_bn,1));
assert(size(d_beta,2) == 1);

d_gamma_zhat = dZ_bn;

d_gamma = sum(d_gamma_zhat .* Z_hat,2);
assert(size(d_gamma,1) == size(dZ_bn,1));
assert(size(d_gamma,2) == 1);

dZ_hat = d_gamma_zhat .* gamma;

divar = sum(dZ_hat .* Z_mu,2);
d_mu1 = dZ_hat .* ivar; 

d_var_sqrt = (-1./(var_sqrt.^2)).*divar;

d_var = (0.5 * 1./sqrt(var+eps)) .* d_var_sqrt;

d_sq = 1/num_training_samples * ones(size(dZ_bn)) .* d_var;

d_mu2 = 2 * Z_mu .* d_sq;

dZ_1 = d_mu1 + d_mu2;

d_mu = -1 * sum(d_mu1 + d_mu2,2);

dZ_2 = 1./num_training_samples .* ones(size(dZ_bn)) .* d_mu;

dZ = dZ_1 + dZ_2;

