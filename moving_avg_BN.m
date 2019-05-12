function [mu_test,sigma_test] = moving_avg_BN(Z,Z_mu,var,mu_test,sigma_test)
beta = 0.9;
num_layers = length(Z);

for i = 1:num_layers
    temp = (Z{i} - Z_mu{i});
    mu{i} = temp(:,1);
    mu_test{i} = beta*mu_test{i} + (1-beta)*mu{i};
    
    sigma{i} = sqrt(var{i});
    sigma_test{i} = beta*sigma_test{i} + (1-beta)*sigma{i};
end 