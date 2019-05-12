function L2_cost = compute_cost_L2_regularisation_part(W,lambda)

m = size(W{1},2);
num_layers = length(W);
L2_cost = 0;
for i = 1:num_layers
    L2_cost = L2_cost + norm(W{i},'fro');
end
L2_cost = lambda * L2_cost / m;