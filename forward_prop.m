function [h_output, prob, loss] = forward_prop(X, y_onehot, W1, b1, W2, b2, lambda)
%% forward propagation for our 1 layer network
%% input parameters
% X is our m x n dataset, where m = number of samples, n = number of
% features
% W1 is our h1 x n weight matrix, where h1 = number of hidden units in
% layer 1
% b1 is the length h1 column vector of bias terms associated with layer 1
% W2 is the c x h1 weight matrix, where c = number of classes
% b2 is the length h2 column vector of bias terms associated with the output
%% output parameters
% returns a probability matrix of dimension m x c, where the element in
% position (i, j) corresponds to the probability that sample i is in class
% j
%% Your code here
% Your code end here
m = size(X,1); % number of samples
n = size(X,2); % number of features
Z1 = W1*X' + repmat(b1, 1, m); % Z1 is h1 x m 
A1 = sigmoid_func(Z1')'; % A1 is h1 x m
Z2 = W2*A1 + repmat(b2, 1, m); % Z2 is c x m
A2 = softmax_func(Z2')'; % A2 is c x m
prob = A2'; % probability matrix is m x c
h_output = A1'; % hidden layer output is m x h1
loss = 1/m*sum(cross_entropy(prob, y_onehot)); % loss for mini-batch
end