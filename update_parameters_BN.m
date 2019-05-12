
function [W,gamma,beta] = update_parameters_BN(W,gamma,beta,learning_rate,dW,d_gamma,d_beta)

num_layers = length(W);

for i = 1:num_layers
        W{i} = W{i} - learning_rate * dW{i};
        gamma{i} = gamma{i} - learning_rate * d_gamma{i};
        beta{i} = beta{i} - learning_rate * d_beta{i};
end