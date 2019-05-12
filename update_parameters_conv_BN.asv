
function [filt,gamma,beta] = update_parameters_conv_BN(filt,gamma,beta,learning_rate,dFilt,dGamma,dBeta)

num_layers = length(filt);

for i = 1:num_layers
        filt{i} = filt{i} - learning_rate * dFilt{i};
        gamma{i} = gamma{i} - learning_rate * dGamma{i};
        beta{i} = beta{i} - learning_rate * dBeta{i};
end