function [dW ,d_gamma, d_beta] = backward_propagation_drop_out_BN...
    (x_train,y_train,A,Z_bn,Z_hat,gamma,Z_mu,ivar,var_sqrt,var,eps,W,activation,prob_matrix,keep_prob_init,keep_prob)
m = size(y_train,2);
num_layers = length(A);
dZ_bn = cell(1,num_layers);
dZ = cell(1,num_layers);
if strcmp(activation{end},'sigmoid')
    dZ_bn{end} = A{end} - y_train;
end

for i = num_layers : -1 : 2
    [dZ{i},d_gamma{i},d_beta{i}] = batch_norm_backward(dZ_bn{i},Z_hat{i},gamma{i}, Z_mu{i}, ivar{i}, var_sqrt{i}, var{i}, eps{i});
    dA{i-1} = (W{i}' * dZ{i}) ;
    dA{i-1} = dA{i-1} .* prob_matrix{i};
    if i == 2
        dA{i-1}  = dA{i-1} ./keep_prob_init;
    else
        dA{i-1}  = dA{i-1} ./keep_prob;
    end
    if strcmp(activation{i-1},'relu')  
        dZ_bn{i-1} = dA{i-1}.* linear_activation_grad(Z_bn{i-1});
    elseif strcmp(activation{i-1},'sigmoid')
        dg = sigmoid_activation_grad(Z_bn{i-1});
        dg(isnan(dg)) = 0;
        dZ_bn{i-1} = dA{i-1} .* dg ;
    end
    
end
    
% i = 1 - 1st layer
i = 1;
    [dZ{i},d_gamma{i},d_beta{i}] = batch_norm_backward(dZ_bn{i},Z_hat{i},gamma{i}, Z_mu{i}, ivar{i}, var_sqrt{i}, var{i}, eps{i});


for i = 1:num_layers
    if i == 1
        dW{i} = 1/m * (dZ{i}*x_train');
    else
        dW{i} = 1/m * (dZ{i}*A{i-1}');
    end
   
end