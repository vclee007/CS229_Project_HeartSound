% ***********************************************
% Implementation of Locally Weighted Logistic Regression
% ***********************************************
close all
clear all

disp('============= Locally Weighted Logistic Regression Model ============');
include_val = 3;

%% Import Files
data_dir = [pwd filesep];
features_file = 'features_train';
label_file = 'classes_train';
features_val_file = 'features_val';
label_val_file = 'classes_val';
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
    X_val = [ones(size(data_val,1), 1), data_val];
    y_val = ytrue_val;
    p = randperm(m);
    m_train = round(m);
    m_dev = size(data_val,1);
    train_idx = p;
    X_dev = X_val;
    y_dev = y_val;
    X_train = [ones(m_train, 1), data(train_idx, :)];
    y_train = ytrue(train_idx);
elseif (include_val == 2) 
    % Mix all data together and then separate into train and val
    X_val = [ones(size(data_val,1), 1), data_val];
    y_val = ytrue_val;
    X_train = [ones(m, 1), data];
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
    X_val = [ones(size(data_val,1), 1), data_val];
    y_val = ytrue_val;
    X_train = [ones(m, 1), data];
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
    X_dev = [ones(m_dev, 1), data(dev_idx, :)];
    y_dev = ytrue(dev_idx);
    X_train = [ones(m_train, 1), data(train_idx, :)];
    y_train = ytrue(train_idx);
end

%% Unit normalization per feature:
if (true)
    X_train = X_train./repmat(sqrt(sum(X_train.^2,1)),m_train,1);
    X_dev = X_dev./repmat(sqrt(sum(X_dev.^2,1)),m_dev,1);
end

%% Prediction 
y_pred = LWLR(X_dev, X_train, y_train, 3);

threshold = 0.5;

% Cross Validation Prediction
y_dev_hat = y_pred > threshold;
y_dev_hat = 2.*y_dev_hat - 1;
dev_err = sum(abs(y_dev - y_dev_hat))/(2*length(y_dev));

% Performance Metric for Dev Set
disp('============= Dev Performance ============');
evaluate_metric(y_dev_hat, y_dev);


%% Locally Weighted Logistic Regression Matlab Function

function y_pred = LWLR(x_predc, x_intc, y_in, tau)
    m = size(x_predc, 1); % num of samples
    y_pred = zeros(m, 1);
    
    % solve for y(i) using x(i)
    for i = 1:m
        w = exp(-sum((x_intc - x_predc(i, :)).^2, 2)./(2.*tau.^2)); %m x 1
        XtWy = transpose(x_intc)*(w.*y_in); % (n x m) * (m x 1) = n x 1
        XtWX = transpose(x_intc)*(w.*x_intc); % (n x m) * (m x n) = n x n
        theta = inv(XtWX)*XtWy; % (n x n) * (n x 1) = n x 1
        y_pred(i) = 1./(1+exp(-x_predc(i, :)*theta));
    end    
end