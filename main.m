%% importing, normalizing and splitting data
pca = 1;
include_val = 3;

data_dir = [pwd filesep];
features_file = 'features_train';
label_file = 'classes_train';
features_val_file = 'features_val';
label_val_file = 'classes_val';

if (0)
    features_file = [features_file, '_pca'];
    features_val_file = [features_val_file, '_pca'];
end

addpath(pwd)
data = dlmread([data_dir features_file '.csv'], ',');
ytrue = dlmread([data_dir label_file '.csv'], ',');
data_val = dlmread([data_dir features_val_file '.csv'], ',');
ytrue_val = dlmread([data_dir label_val_file '.csv'], ',');
n = size(data, 2);
m = size(data, 1);

% Ratio of training examples
if (include_val == 1)
    % Use train and validation datasets
    X_val = data_val;
    y_val = ytrue_val;
    p = randperm(m);
    m_train = round(m);
    m_dev = size(data_val,1);
    train_idx = p;
    X_dev = X_val;
    y_dev = y_val;
    X_train = data(train_idx, :);
    y_train = ytrue(train_idx);
elseif (include_val == 2) 
    % Mix all data together and then separate into train and val
    X_val = data_val;
    y_val = ytrue_val;
    X_train = data;
    y_train = ytrue;
    X_train_all = [X_train; X_val];
    y_train_all = [y_train; y_val];
    m_train_all = size(X_train_all, 1);
    
    r = 0.8;
    p = randperm(m_train_all);
    m_train = round(m_train_all*r);
    m_dev = m_train_all - m_train;
    train_idx = p(1:m_train);
    dev_idx = p(m_train+1:end);
    X_dev = X_train_all(dev_idx, :);
    y_dev = y_train_all(dev_idx);
    X_train = X_train_all(train_idx, :);
    y_train = y_train_all(train_idx);
elseif (include_val == 3) 
    % Mix all data together and then separate into train and val 
    % Both train and val have 50% mix of classes
    X_val = data_val;
    y_val = ytrue_val;
    X_train = data;
    y_train = ytrue;
    X_train_all = [X_train; X_val];
    y_train_all = [y_train; y_val];
    m_train_all = size(X_train_all, 1);
    r = 0.8;
    
    idx_p = find(y_train_all == 1);
    idx_n = find(y_train_all == -1);
    m_balanced = min(length(idx_p), length(idx_n));
    p_p = randperm(m_balanced);
    p_n = randperm(length(idx_n));
    m_train_p = round(m_balanced * r);
    m_dev_p = m_balanced - m_train_p;
    train_idx_p = idx_p(p_p(1:m_train_p));
    train_idx_n = idx_n(p_n(1:m_train_p));
    train_idx = [train_idx_p; train_idx_n];
    train_idx = train_idx(randperm(length(train_idx)));
    X_train = X_train_all(train_idx, :);
    y_train = y_train_all(train_idx, 1);
    m_train = size(X_train, 1);
    
    dev_idx_p = idx_p(p_p(m_train_p+1:m_train_p + m_dev_p));
    dev_idx_n = idx_n(p_n(m_train_p+1:m_train_p + m_dev_p));
    dev_idx = [dev_idx_p; dev_idx_n];
    dev_idx = dev_idx(randperm(length(dev_idx)));
    X_dev = X_train_all(dev_idx, :);
    y_dev = y_train_all(dev_idx, 1);
    m_dev = size(X_dev, 1);
else
    % Use only train dataset for both train and validation
    r = 0.7;
    p = randperm(m);
    m_train = round(m*r);
    m_dev = m - m_train;
    train_idx = p(1:m_train);
    dev_idx = p(m_train+1:end);
    X_dev = data(dev_idx, :);
    y_dev = ytrue(dev_idx);
    X_train = data(train_idx, :);
    y_train = ytrue(train_idx);
end

m_train_size = m_train;
m_test = m_dev;
Xtrain = X_train;
Xtest = X_dev;
ytrain = y_train;
ytest = y_dev;

if(0)
    data_dir = [pwd filesep];
    features_file = 'features_train';
    label_file = 'classes_train';
    features_dev_file = 'features_val';
    label_dev_file = 'classes_val';
    addpath(pwd)
    data = dlmread([data_dir features_file '.csv'], ',');
    ytrue = dlmread([data_dir label_file '.csv'], ',');
    data_test = dlmread([data_dir features_dev_file '.csv'], ',');
    ytrue_test = dlmread([data_dir label_dev_file '.csv'], ',');

    n = size(data, 2);
    m_all = size(data, 1);

    % Ratio of training examples
    r = 0.8;
    p = randperm(m_all);
    m_train = round(m_all*r);
    m_dev = m_all - m_train;
    train_idx = p(1:m_train);
    dev_idx = p(m_train+1:end);
end
% adding intercept to data X
train_data = X_train; 
train_label = y_train;
test_data = X_dev;
test_label = y_dev;

%% importing, normalizing and splitting data
rng(20171115, 'twister');
% if (exist('train_data', 'var') ~= 1)
%     train_data = load('images_train.csv');
%     train_label = load('labels_train.csv');
%     test_data = load('images_test.csv');
%     test_label = load('labels_test.csv');
% end

r = 0.9;
n = size(train_data, 2);
c = 2;
train_size = round(r*m_train);
p = randperm(size(train_data, 1));

% converting labels to one hot encoding
y_train = zeros(c, size(train_label, 1));
y_train((train_label + 1)./2 + 1 + c * (0:size(train_label, 1)-1)') = 1;
y_train = y_train';

y_test = zeros(c, size(test_label, 1));
y_test((test_label + 1)./2 + 1 + c * (0:size(test_label, 1)-1)') = 1;
y_test = y_test';

X_train_all = train_data(p, :);
y_train_all = y_train(p, :);
X_train = X_train_all(1:train_size, :);
X_dev = X_train_all(train_size+1:end, :);
y_train = y_train_all(1:train_size, :);
y_dev = y_train_all(train_size+1:end, :);
X_test = test_data;

%normalize the data
if (0)
    avg = mean(mean(X_train));
    s = std(reshape(X_train, [], 1));

    X_train = (X_train - avg) / s;
    X_dev = (X_dev - avg) / s;
    X_test = (X_test - avg) / s;
end
    
if (true)
    X_train = (X_train - mean(X_train))./std(X_train);
    X_dev = (X_dev - mean(X_dev))./std(X_dev);
    X_test = (X_test - mean(X_test))./std(X_test);
end


%% training the network (without regularization)
m = train_size;
h1 = 20; % 300 units in hidden layer

% initialize the parameters
W1 = randn(h1, n);
b1 = zeros(h1, 1);
W2 = randn(c, h1);
b2 = zeros(c, 1);

num_epoch = 50;
batch_size = 50;
num_batch = m / batch_size;
learning_rate = 5;
lambda = 0.001;
train_loss = zeros(num_epoch, 1);
dev_loss = zeros(num_epoch, 1);
train_accuracy = zeros(num_epoch, 1);
dev_accuracy = zeros(num_epoch, 1);

%% Your code here
% num_epoch = 5;
% batch_size = 1;
% num_batch = m/batch_size;
% train_loss = zeros(num_epoch, 1);
% dev_loss = zeros(num_epoch, 1);
% train_accuracy = zeros(num_epoch, 1);
% dev_accuracy = zeros(num_epoch, 1);

for epoch = 1:num_epoch
    learning_rate = learning_rate/(epoch).^1.0;
    for batch = 0:num_batch-1
        X_batch = X_train(1+batch_size*batch:batch_size*(batch+1),:);
        y_batch = y_train(1+batch_size*batch:batch_size*(batch+1),:);
        [h_output, prob, loss] = forward_prop(X_batch, y_batch, W1, b1, W2, b2, lambda);
        [dW1, db1, dW2, db2] = backward_prop(X_batch, y_batch, W1, b1, W2, b2, lambda);
        W1 = W1 - learning_rate.*dW1;
        b1 = b1 - learning_rate.*db1;
        W2 = W2 - learning_rate.*dW2;
        b2 = b2 - learning_rate.*db2;
    end
    [h_train, train_prob, train_loss_] = forward_prop(X_batch, y_batch, W1, b1, W2, b2, lambda);
    [h_dev, dev_prob, dev_loss_] = forward_prop(X_dev, y_dev, W1, b1, W2, b2, lambda);
    train_loss(epoch,1) = train_loss_;
    dev_loss(epoch,1) = dev_loss_;
    [y_train_hat, i_train] = max(train_prob,[],2);
    [y_train_gt, i_train_gt] = max(y_batch,[],2);
    [y_dev_hat, i_dev] = max(dev_prob,[],2);
    [y_dev_gt, i_dev_gt] = max(y_dev,[],2);
    train_accuracy(epoch,1) = sum(i_train == i_train_gt)/batch_size;
    dev_accuracy(epoch,1) = sum(i_dev == i_dev_gt)/size(y_dev,1);
end
save params_reg W1 b1 W2 b2
% Your code end here
%% plotting and displaying results
figure(1);
plot(1:num_epoch, train_loss, 'r', ...
    1:num_epoch, dev_loss, 'b');
legend('training set', 'dev set');
xlabel('epochs');
ylabel('loss');
figure(2);
plot(1:num_epoch, train_accuracy, 'r', ...
    1:num_epoch, dev_accuracy, 'b');
legend('training set', 'dev set');
xlabel('epochs');
ylabel('accuracy');

%% Testing the model 
% load params
load params_reg

% Training Evaluation
[~, train_pred, ~] = forward_prop(X_train, y_train, W1, b1, W2, b2, lambda);
[~, y_c_train_pred] = max(train_pred, [], 2);
[~, y_c_train] = max(y_train, [], 2);
train_accuracy = sum(y_c_train_pred == y_c_train) / size(y_train, 1);
fprintf('training set accuracy: %f \n', train_accuracy);

pred_train = (y_c_train_pred - 1).*2 - 1;
actual_train = (y_c_train - 1).*2 -1;
disp('============= Train Performance =============')
evaluate_metric(pred_train, actual_train);

% Cross Validation
[~, test_pred, ~] = forward_prop(X_test, y_test, W1, b1, W2, b2, lambda);
[~, y_c_test_pred] = max(test_pred, [], 2);
[~, y_c_test] = max(y_test, [], 2);
test_accuracy = sum(y_c_test_pred == y_c_test) / size(y_test, 1);
fprintf('test set accuracy: %f \n', test_accuracy);

pred = (y_c_test_pred - 1).*2 - 1;
actual = (y_c_test - 1).*2 -1;
disp('============= Test Performance =============')
evaluate_metric(pred, actual);

