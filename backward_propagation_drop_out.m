function [dW ,db] = backward_propagation_drop_out(x_train,y_train,A,Z,W,activation,prob_matrix,keep_prob_init,keep_prob)
m = size(y_train,2);
num_layers = length(A);
dZ = cell(1,num_layers);
if strcmp(activation{end},'sigmoid')
    dZ{end} = A{end} - y_train;
end

for i = num_layers : -1 : 2
    dA{i-1} = (W{i}' * dZ{i}) ;
    dA{i-1} = dA{i-1} .* prob_matrix{i};
    if i == 2
        dA{i-1}  = dA{i-1} ./keep_prob_init;
    else
        dA{i-1}  = dA{i-1} ./keep_prob;
    end
    if strcmp(activation{i-1},'relu')  
        dZ{i-1} = dA{i-1}.* linear_activation_grad(Z{i-1});
    elseif strcmp(activation{i-1},'sigmoid')
        dg = sigmoid_activation_grad(Z{i-1});
        dg(isnan(dg)) = 0;
        dZ{i-1} = dA{i-1} .* dg ;
    end
    
end

for i = 1:num_layers
    if i == 1
        dW{i} = 1/m * (dZ{i}*x_train');
    else
        dW{i} = 1/m * (dZ{i}*A{i-1}');
    end
    
    db{i} = 1/m * sum(dZ{i},2);
end