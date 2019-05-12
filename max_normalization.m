function W_normalized = max_normalization(W,c)

num_layers = length(W);
for i = 1:num_layers
    W_norm(i) = norm(W{i},'fro');
    if W_norm(i) < c
        W_normalized{i} = W{i};
        continue;
    else
        W_normalized{i} = (W{i}*c)/W_norm(i);
    end
end

end