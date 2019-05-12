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
learning_rate = 0.005;  
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
num_iter = 30;
% Enable max norm 
max_norm = 0;
% Optimisation method
opt = 'momentum';

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
[W, b] = initialize_parameters(layer_dims);
W_intial = W;
b_initial = b;
%% Keep_prob value
keep_prob = 0.5;
keep_prob_init = 0.8;
%% Initialize V_dw and V_db for gradient descent with momentum
[V_dW, V_db] = initialize_parameters_with_zero(layer_dims);
%% check that size of W and V_dW are same for every layer. Same thing for b
for i = 1:num_layers
    [a1 , a2] = size(W{i});
    [c1 ,c2] = size(V_dW{i});
    assert(a1 == c1);
    assert(a2 == c2);
end

for i = 1:num_layers
    [a1 , a2] = size(b{i});
    [c1 ,c2] = size(V_db{i});
    assert(a1 == c1);
    assert(a2 == c2);
end
%% Initialize S_dw and S_db for gradient descent with RMS prop
[S_dW, S_db] = initialize_parameters_with_zero(layer_dims);
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
for iter = 1 : num_iter
    if mod(iter,3) == 0
        disp(['iteration number ', num2str(iter),' started'])
    end
    for k = 1 : length(x_train_mini_batch)
        
        
        if drop_out
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
            [A,Z] = forward_propagation_drop_out(x_train_mini_batch{k},W,b,activation,prob_matrix,keep_prob_init,keep_prob);
            %% Computing cost from last layer activation
            cost(k,iter) = compute_cost(y_train_mini_batch{k},A);
            if isnan(cost(k,iter))
                break;
            end
            %% Backward propagation
            [dW ,db] = backward_propagation_drop_out(x_train_mini_batch{k},y_train_mini_batch{k},A,Z,W,activation,prob_matrix,keep_prob_init,keep_prob);
        else
            [A,Z] = forward_propagation(x_train_mini_batch{k},W,b,activation);
            cost(k,iter) = compute_cost(y_train_mini_batch{k},A) + compute_cost_L2_regularisation_part(W,lambda);
            if isnan(cost(iter))
                break;
            end
            [dW, db] = backward_propagation(x_train_mini_batch{k},y_train_mini_batch{k},A,Z,W,activation);
            dW = back_prop_L2(dW,W,lambda);
        end
        %% Update Parameters - normal gradient descent
        if strcmp(opt,'normal') == 1
            [W,b] = update_parameters(W, b, learning_rate, dW, db);
        elseif strcmp(opt,'momentum') == 1
            %% Update Parameters with momentum
            [V_dW, V_db] =  momentum_avg(V_dW,dW,V_db,db,iter);
            [W,b] = update_parameters(W,b,learning_rate,V_dW,V_db);
        elseif strcmp(opt,'RMS') == 1
            %% Update Parameters with RMS Prop
                [S_dW, S_db] = rmsProp_avg(S_dW,dW,S_db,db,iter);
                [W,b] = update_parameters_rmsProp(W,b,learning_rate,dW,db,S_dW,S_db);
        elseif strcmp(opt,'adam') == 1
            %% Update Parameters with Adam
                [V_dW, V_db] =  momentum_avg(V_dW,dW,V_db,db,iter);
                [S_dW, S_db] = rmsProp_avg(S_dW,dW,S_db,db,iter);
                [W,b] = update_parameters_rmsProp(W,b,learning_rate,V_dW,V_db,S_dW,S_db);
        end
        
        if max_norm == 1
            W = max_normalization(W,c);
            b = max_normalization(b,c);
        end
        %% end the training loop
    end
end


%% plot cost
cost = cost(:);
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
[A_main,~] = forward_propagation(x_test,W,b,activation);
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
