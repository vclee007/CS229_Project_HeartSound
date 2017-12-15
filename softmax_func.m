function prob = softmax_func(X)
%% softmax function
%% input parameters: x = m x c matrix, m is the block size, 
% n is the number of classes
%% output parameters: : m x c matrix
%% Your code here
% Your code end here
prob = exp(X)./sum(exp(X),2);
end
