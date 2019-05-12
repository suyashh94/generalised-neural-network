function [V_dW, V_dGamma, V_dBeta] = momentum_avg_BN(V_dW,dW,V_dGamma,d_gamma,V_dBeta,d_beta,count)
beta1 = 0.8;
num_layers = length(dW);
for i = 1:num_layers
V_dW{i} = beta1 * V_dW{i} + (1-beta1) * dW{i};
% V_dW{i} = V_dW{i}/(1-beta1^count);
V_dGamma{i} = beta1 * V_dGamma{i} + (1-beta1) * d_gamma{i};
% V_dGamma{i} = V_dGamma{i}/(1-beta1^count);
V_dBeta{i} = beta1 * V_dBeta{i} + (1-beta1) * d_beta{i};
% V_dBeta{i} = V_dBeta{i}/(1-beta1^count);

end 
end 