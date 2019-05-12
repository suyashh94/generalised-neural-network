function [W,beta,gamma,V_dW,V_dBeta,V_dGamma,S_dW,S_dBeta,S_dGamma] = adam_update_BN(V_dW,dW,V_dBeta,d_beta,V_dGamma,d_gamma,S_dW,S_dBeta,S_dGamma,count,W,beta,gamma,learning_rate)
beta1 = 0.9;
beta2= 0.9;
num_layers = length(dW);
eps = 10^-8;
for i = 1:num_layers
V_dW{i} = beta1 * V_dW{i} + (1-beta1) * dW{i};
V_dW{i} = V_dW{i}./(1-beta1^count);
V_dBeta{i} = beta1 * V_dBeta{i} + (1-beta1) * d_beta{i};
V_dBeta{i} = V_dBeta{i}./(1-beta1^count);
V_dGamma{i} = beta1 * V_dGamma{i} + (1-beta1) * d_gamma{i};
V_dGamma{i} = V_dGamma{i}./(1-beta1^count);

S_dW{i} = beta2 * S_dW{i} + (1-beta2) * dW{i}.^2;
S_dW{i} = S_dW{i}./(1-beta2^count);

S_dBeta{i} = beta2 * S_dBeta{i} + (1-beta2) * d_beta{i}.^2;
S_dBeta{i} = S_dBeta{i}./(1-beta2^count);

S_dGamma{i} = beta2 * S_dGamma{i} + (1-beta2) * d_gamma{i}.^2;
S_dGamma{i} = S_dGamma{i}./(1-beta2^count);

W{i} = W{i} - learning_rate .* (V_dW{i}./ (sqrt(S_dW{i}) + eps ));
beta{i} = beta{i} - learning_rate .* (V_dBeta{i}./ (sqrt(S_dBeta{i}) + eps ));
gamma{i} = gamma{i} - learning_rate .* (V_dGamma{i}./ (sqrt(S_dGamma{i}) + eps ));

end 
end 