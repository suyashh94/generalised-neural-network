clear all;
%% Load training data
load('C:\Users\harla\Desktop\Matlab scripts\Datasets\MNIST Fashion\training_set.mat');
%% Normalize training data - here just diving by 255
% x_train = im2double(x_train);
x_train = x_train(:,1:60000);
y_train = y_train(:,1:60000);
y_train_main = zeros(max(y_train)+1,size(x_train,2));
for i = 1:length(y_train)
    y_train_main(y_train(i)+1,i) = 1;
end
y_train = y_train_main;
t = shuffle(1:60000);
x_train = x_train(:,t(1:60000));
y_train = y_train(:,t(1:60000));
%% Constant terms
% x_train = x_train/255;

% Learning rate
learning_rate = 0.1;
% Weight projection sphere
c = 3;
% Regularisation constant
lambda = 0;
% Dropout
drop_out = 1;
if drop_out == 1
    lambda = 0;
end
% Number of iterations
num_iter = 100;
% Enable max norm
max_norm = 0;
% Optimisation method
opt = 'adam';

clear y_train_main

%% Show the network traits

if drop_out == 1
    disp('Dropout enabled')
else
    disp('Dropout disabled')
end

if max_norm == 1
    disp('Max norm enabled')
else
    disp('max norm disabled')
end

if lambda > 0
    disp('regularisation enabled')
else
    disp('regularisation disabled')
end

disp(['learning rate is ' , num2str(learning_rate)]);
disp(['Optimization algorithm is ', opt])
%% Make batches of training set

batch_size = 10;
batch_num = size(x_train,2)/batch_size;
for k = 1:batch_num
    x_train_mini_batch{k} = x_train(:,(k-1)*batch_size + 1 : k*batch_size);
    y_train_mini_batch{k} = y_train(:,(k-1)*batch_size + 1 : k*batch_size);
end

%% Number of layers
num_layers = input('Enter number of layers excluding input layer ');
layer_dims = zeros(num_layers+1,1);
layer_dims(1) = size(x_train,1);
layer_dims(end) = size(y_train,1);
for i = 2:length(layer_dims)-1
    layer_dims(i) = input(['Enter number of units in layer ' num2str(i)]);
end
%% Initialize weights of all hidden layers
[W, ~] = initialize_parameters(layer_dims);
[gamma , beta] = initialize_gamma_and_beta(layer_dims);
W_intial = W;
%% Initialize mu and sigma for test
for i = 1:num_layers
    mu_test{i} = zeros(layer_dims(i+1),1);
    sigma_test{i} = zeros(layer_dims(i+1),1);
end
%% Keep_prob value
keep_prob = 0.5;
keep_prob_init = 0.8;
%% Initialize V_dw and , V_dGamma, V_dBeta for gradient descent with momentum
[V_dW, ~] = initialize_parameters_with_zero(layer_dims);
[V_dGamma, V_dBeta] = initialize_parameters_with_zero_beta_gamma(layer_dims);
%% Initialize S_dw and , S_dGamma, S_dBeta for gradient descent with RMS prop
[S_dW, ~] = initialize_parameters_with_zero(layer_dims);
[S_dGamma, S_dBeta] = initialize_parameters_with_zero_beta_gamma(layer_dims);

%% beta1 and beta 2 dissapearance over counts
beta1 = 0.9; 
beta2 = 0.9;
%% check that size of W and V_dW are same for every layer. Same thing for b
for i = 1:num_layers
    [a1 , a2] = size(W{i});
    [c1 ,c2] = size(V_dW{i});
    assert(a1 == c1);
    assert(a2 == c2);
end
%% Defining activations for every layer
% Last layer has activation sigmoid
% All layers before it has activation relu
for i = 1:num_layers
    if i ~= num_layers
        if mod(i,2) > -1
            activation{i} = 'relu';
        else
            activation{i} = 'sigmoid';
        end
    else
        activation{i} = 'sigmoid';
    end
end

%% Start the training loop
count = 0;
for iter = 1 : num_iter
    if mod(iter,3) == 0
        disp(['iteration number ', num2str(iter),' started'])
    end
    for k = 1: length(x_train_mini_batch)
        
        
        for layer = 1:length(layer_dims)
            prob_matrix{layer} = rand(layer_dims(layer), batch_size);
            if layer == 1
                prob_matrix{layer} = prob_matrix{layer} < keep_prob_init;
            elseif layer == length(layer_dims)
                prob_matrix{layer} = prob_matrix{layer} > -1;    % Everything equal to 1 for output layer
            else
                prob_matrix{layer} = prob_matrix{layer} < keep_prob;
            end
        end
        %% Forward propagation of the network
        [A,Z,Z_bn,Z_hat,gamma, Z_mu, ivar, var_sqrt, var, eps] ...
            = forward_propagation_drop_out_BN(x_train_mini_batch{k},W,gamma,beta,activation,prob_matrix,keep_prob_init,keep_prob)  ;          %% Computing cost from last layer activation
        cost(k,iter) = compute_cost(y_train_mini_batch{k},A);
        
        if isnan(cost(k,iter))
            break;
        end
        
        [mu_test,sigma_test] = moving_avg_BN(Z,Z_mu,var,mu_test,sigma_test);
        %% Backward propagation
        [dW ,d_gamma, d_beta] = backward_propagation_drop_out_BN...
            (x_train_mini_batch{k},y_train_mini_batch{k},A,Z_bn,Z_hat,gamma,Z_mu,ivar,var_sqrt,var,eps,W,activation,prob_matrix,keep_prob_init,keep_prob);
        
        
        %% Update Parameters - normal gradient descent
        if strcmp(opt,'normal') == 1
            [W,gamma,beta] = update_parameters_BN(W,gamma,beta,learning_rate,dW,d_gamma,d_beta);
        elseif strcmp(opt,'momentum') == 1
            %% Update Parameters with momentum
            count = count + 1;
            [V_dW, V_dGamma, V_dBeta] = momentum_avg_BN(V_dW,dW,V_dGamma,d_gamma,V_dBeta,d_beta,count);
            [W,gamma,beta] = update_parameters_BN(W,gamma,beta,learning_rate,V_dW,V_dGamma,V_dBeta);
            
        elseif strcmp(opt,'RMS') == 1
            %% Update Parameters with RMS Prop
            count = count + 1;
            [S_dW, S_db] = rmsProp_avg(S_dW,dW,S_db,db,count);
            [W,b] = update_parameters_rmsProp(W,b,learning_rate,dW,db,S_dW,S_db);
        elseif strcmp(opt,'adam') == 1
            %% Update Parameters with Adam
            count = count + 1;
            [W,beta,gamma,V_dW,V_dBeta,V_dGamma,S_dW,S_dBeta,S_dGamma] = ...
                adam_update_BN(V_dW,dW,V_dBeta,d_beta,V_dGamma,d_gamma,S_dW,S_dBeta,S_dGamma,count,W,beta,gamma,learning_rate);        
            beta1_counter(count) = (1-beta1^count);
            beta2_counter(count) = (1-beta2^count);
        end
        
        if max_norm == 1
            W = max_normalization(W,c);
        end
        %% end the training loop
    end
end


%% plot cost
cost = mean(cost,2);
figure;

plot(cost)
%% plot A
figure;
imagesc(A{end})

%% Load x_test
load('C:\Users\harla\Desktop\Matlab scripts\Datasets\MNIST Fashion\test_set.mat');
% x_test = im2double(x_test);
%% Making y test for mnist fashion
y_test_main = zeros(size(y_train,1),size(x_test,2));
for i = 1:length(y_test)
    y_test_main(y_test(i)+1,i) = 1;
end
y_test = y_test_main;
%%
% x_test = x_train;
% y_test = y_train;
%% Predict y
y_predict = zeros(size(y_train,1),size(x_test,2));
[A_main,~] = forward_propagation_BN_Test(x_test,W,gamma,beta,activation,mu_test,sigma_test);
A_predict = A_main{end};
for i = 1:size(A_predict,2)
    [~,dummy] = max(A_predict(:,i));
    y_predict(dummy,i) = 1;
end
%%

figure;
hold on
subplot(2,1,1)
imagesc(y_test)

subplot(2,1,2)
imagesc(y_predict)

%% error

error = (sum(abs(y_test - y_predict)));
error = length(find(error > 0))/ size(y_test,2);

%% Confusion Matrix
for i = 1:size(y_test,2)
    y_predict_val(i) = find(y_predict(:,i) == 1);
end

for i = 1:size(y_test,2)
    y_test_val(i) = find(y_test(:,i) == 1);
end

C = confusionmat(y_test_val, y_predict_val);

figure;
imagesc(C);

%%
clear x_test y_test
