function [A,Z,Z_bn,Z_hat,gamma, Z_mu, ivar, var_sqrt, var, eps] ...
    = forward_propagation_BN_Test(x_test,W,gamma,beta,activation,mu_test,sigma_test)
% This does the forward propagation
% Input are x_train in n_feature X n_training_sample
% W,b and activation method for each layer - relu or sigmoid
% Returns A and Z for every layer
num_training_samples = size(x_test,2);
num_layers = length(W);
assert(length(activation) == num_layers);

A_prev = x_test;

for i = 1:num_layers
    Z{i} = W{i} * A_prev ;
    [Z_bn{i},Z_hat{i},gamma{i}, Z_mu{i}, ivar{i}, var_sqrt{i}, var{i}, eps{i}]...
        = batch_norm_forward_Test(Z{i},gamma{i},beta{i},mu_test{i},sigma_test{i});
    if strcmp(activation{i},'relu')
        A{i} = linear_activation(Z_bn{i});
        assert(size(A{i},2) == num_training_samples);
    elseif strcmp(activation{i},'sigmoid')
        A{i} = sigmoid_activation(Z_bn{i});
        assert(size(A{i},2) == num_training_samples);
    end
    A_prev = A{i};
end