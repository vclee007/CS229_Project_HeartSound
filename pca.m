data_dir = [pwd filesep];
features_file = 'features_train';
features_val_file = 'features_val';
if (1)
    features_file = features_val_file;
end
addpath(pwd)
fn = dir([data_dir features_file '.csv']);
data = dlmread([data_dir features_file '.csv'], ',');

n = size(data, 2);
m = size(data, 1);

X = data(:, :);


%% Unit normalization per feature:
if (true)
    X = X - mean(X);
    X = X ./ std(X);
end

%% Principal Components Analysis

% Gram matrix and Eigendecomposition
sig = 1/m.*X'*X;
[V, D] = eig(sig);

% k - new (reduced) dimension, and u is n x k 
k = 15;
u = V(:, 1:k);

% new features mapped from n x m to k x m
features = X * u;

% Save features to CSV file
output_file = [features_file, '_pca'];

addpath(pwd)
dlmwrite([data_dir output_file '.csv'], features, 'delimiter',',');
