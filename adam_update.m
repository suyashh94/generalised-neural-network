function [W,b,V_dW,V_db,S_dW,S_db] = adam_update(V_dW,dW,V_db,db,S_dW,S_db,count,W,b,learning_rate)
beta1 = 0.2;
beta2= 0.2;
num_layers = length(dW);
eps = 10^-8;
for i = 1:num_layers
V_dW{i} = beta1 * V_dW{i} + (1-beta1) * dW{i};
V_dW{i} = V_dW{i}./(1-beta1^count);
V_db{i} = beta1 * V_db{i} + (1-beta1) * db{i};
V_db{i} = V_db{i}./(1-beta1^count);


S_dW{i} = beta2 * S_dW{i} + (1-beta2) * dW{i}.^2;

S_dW{i} = S_dW{i}./(1-beta2^count);

S_db{i} = beta2 * S_db{i} + (1-beta2) * db{i}.^2;

S_db{i} = S_db{i}./(1-beta2^count);


W{i} = W{i} - learning_rate .* (V_dW{i}./ (sqrt(S_dW{i}) + eps ));
b{i} = b{i} - learning_rate .* (V_db{i}./ (sqrt(S_db{i}) + eps ));

end 
end 