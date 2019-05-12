function [S_dW, S_db] = rmsProp_avg(S_dW,dW,S_db,db,iter)
beta = 0.0;
num_layers = length(dW);
for i = 1:num_layers
S_dW{i} = beta * S_dW{i} + (1-beta) * dW{i}.^2;
S_dW{i} = S_dW{i}/(1-beta^iter);
S_db{i} = beta * S_db{i} + (1-beta) * db{i}.^2;
S_db{i} = S_db{i}/(1-beta^iter);
end 
end 