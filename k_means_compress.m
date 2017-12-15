clear all;
close all;

%% Read in data
include_val = 3;

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

n_all = size(data,2);
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

if (0)
    data_dir = [pwd filesep];
    features_file = 'features_train';
    label_file = 'classes_train';
    addpath(pwd)
    data = dlmread([data_dir features_file '.csv'], ',');
    ytrue = dlmread([data_dir label_file '.csv'], ',');
    m_all = size(data,1);
    n_all = size(data,2);

    %% Split the data into train and test sets:

    % training example ratio
    r_train = 0.8;
    m_train = round(r_train*m_all);

    % randomize order of dataset
    data_indices = randperm(m_all, m_all); 

    % pick out r_train*m_all of m_all examples for training
    train_index = data_indices(1:m_train);
    Xtrain = data(train_index,:);
    ytrain = ytrue(train_index);

    % pick out remaining examples for testing segmentation
    test_index = data_indices(m_train+1:end);
    Xtest = data(test_index,:);
    ytest = ytrue(test_index);
    m_test = m_all - m_train; %leftover is test
end
    
% Error Analysis Flag
error_analysis = 1;
lrhsmm_mode = 1;
if(lrhsmm_mode == 1) 
    features_file_gt = 'features_lrhsmm'; 
else
    features_file_gt = 'features_gt';
end
label_file_lrhsmm = 'classes_lrhsmm'; %label for dev
if (error_analysis)
    data_gt = dlmread([data_dir features_file_gt '.csv'], ',');
    ytrue_lrhsmm = dlmread([data_dir label_file_lrhsmm '.csv'], ',');
    Xtest = data_gt;
    ytest = ytrue_lrhsmm;
    m_test = size(ytest, 1);
end

%% Unit normalization per feature:
if (true)
    Xtrain = Xtrain./repmat(sqrt(sum(Xtrain.^2,1)),m_train,1);
    Xtest = Xtest./repmat(sqrt(sum(Xtest.^2, 1)), m_test, 1);
end

%% K-means clustering
% k = number of clusters
% Xtrain = Xtest;
% ytrain = ytest;
k = 2;
m = n_all;
max_iter = 1000;

init_pix = init_clusters(k, m_train);
mus = Xtrain(init_pix,:);
c = assignCluster(Xtrain, mus);
J = calcDistortion(Xtrain, mus, c);
J_prev = J;
converge_count = 0;

figure(3); hold on;
for iter = 1:max_iter
    % set clusters of every pixel to minimum of square distance from centroid
    c = assignCluster(Xtrain, mus);
    
    % recalculate cluster centroids
    mus = updateCentroids(Xtrain, c);
   
    % calculate distortion function
    J = calcDistortion(Xtrain, mus, c);
    delta = J_prev-J;
    if (delta < 1e-10)
        fprintf('Convergence reached\n');
        converge_count = converge_count + 1;
        if (converge_count >= 10)
            break;
        end
    end
    J_prev = J;
    % plot mus
    figure(3); scatter(iter, J); 
end
figure(3); hold off;

ytrain_hat = (2.*(2-c)-1)';
error1 = (100*sum(ytrain_hat .* ytrain <= 0) / length(ytrain));
errorm1 = (100*sum(-1.*ytrain_hat .* ytrain <= 0) / length(ytrain));
train_err = min(error1, errorm1);
fprintf(1, 'Train error rate: %1.4f\n', train_err);

if (errorm1 < error1)
    ytrain_hat = -1.*ytrain_hat;
end

disp('============= Train Performance =============');
evaluate_metric(ytrain_hat, ytrain);

%% Prediction on Test Set
pred_c = assignCluster(Xtest, mus);
ytest_hat = (2.*(2-pred_c)-1)';
error1_test = (100*sum(ytest_hat .* ytest <= 0) / length(ytest));
errorm1_test = (100*sum(-1.*ytest_hat .* ytest <= 0) / length(ytest));
test_err = min(error1_test, errorm1_test);
fprintf(1, 'Test error rate: %1.4f\n', test_err);

if (errorm1_test < error1_test)
    ytest_hat = -1.*ytest_hat;
end

disp('============= Test Performance =============');
evaluate_metric(ytest_hat, ytest);


%% Cluster centroid initialization
function pidx = init_clusters(k, m)
% Initialize clusters by selecting random pixels
% k - number of clusters
% m - number of pixels (examples)
%     pidx = round(1+rand(k,1).*(m-1));
    pidx = randperm(m, k)';
end

%% Assigning every pixel to one of the clusters
function c = assignCluster(x, mus)
    m = size(x,1);
    for ii = 1:m 
       [minval, c(ii)] = min(sum((x(ii,:)-mus(:,:)).^2,2));
    end
end

%% Recalculate cluster centroids
function mus = updateCentroids(x, c)
    m = size(x);
    k = max(c);
    for ii = 1:k
        indx = find(c == ii);
        m = length(indx);
        mus(ii,:) = sum(x(indx,:),1)./m;
    end
end

%% Calculate distortion function J(c,mu)
function J = calcDistortion(x, mus, c)
    m = size(x,1);
    J = sum(sum((x(:,:) - mus(c,:)).^2,2),1);
end