function [V_dW, V_db] = momentum_avg(V_dW,dW,V_db,db,count)
beta = 0.9;
num_layers = length(dW);
for i = 1:num_layers
V_dW{i} = beta * V_dW{i} + (1-beta) * dW{i};
% V_dW{i} = V_dW{i}/(1-beta^count);
V_db{i} = beta * V_db{i} + (1-beta) * db{i};
% V_db{i} = V_db{i}/(1-beta^count);
end 
end 