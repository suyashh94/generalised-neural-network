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
x_train = x_train(:,t);
y_train = y_train(:,t);
% x_train = x_train/255;
learning_rate = 0.6;  % Increase it
clear y_train_main
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
for iter = 1 : 500
    iter
    %% Forward propagation of the network
    [A,Z] = forward_propagation(x_train,W,b,activation);
    %% Computing cost from last layer activation
    cost(iter) = compute_cost(y_train,A);
    if isnan(cost(iter)) 
        break;
    end
    %% Backward propagation
    [dW ,db] = backward_propagation(x_train,y_train,A,Z,W,activation);
    %% Update Parameters - normal gradient descent
    %     [W,b] = update_parameters(W, b, learning_rate, dW, db);
    %% Update Parameters with momentum
    [V_dW, V_db] =  momentum_avg(V_dW,dW,V_db,db,iter);
    [W,b] = update_parameters(W,b,learning_rate,V_dW,V_db);
    %% Update Parameters with RMS Prop
%     [S_dW, S_db] = rmsProp_avg(S_dW,dW,S_db,db,iter);
%     [W,b] = update_parameters_rmsProp(W,b,learning_rate,dW,db,S_dW,S_db);
    %% Update Parameters with Adam
%     [V_dW, V_db] =  momentum_avg(V_dW,dW,V_db,db,iter);
%     [S_dW, S_db] = rmsProp_avg(S_dW,dW,S_db,db,iter);
%     [W,b] = update_parameters_rmsProp(W,b,learning_rate,V_dW,V_db,S_dW,S_db);
    %% end the training loop
end


%% plot cost
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
error = length(find(error > 0)) / size(y_test,2);


