# CS229_Project_HeartSound
Application of Machine Learning Techniques for Heart Sound Recording Classification By Vincent Lee and Anatoly Yakovlev

Dataset and LR-HSMM starter code were provided by Physionet: https://www.physionet.org/challenge/2016/

The dataset is not included in this repo.  Please download them separately from https://www.physionet.org/physiobank/database/challenge/2016/ 

Also, example_data.mat training data for LR-HSMM can be downloaded from https://www.physionet.org/physiotools/hss/

To run only classification on supplied features follow these steps:
1. For logistic regression run: lr_model.m
2. For weighted logistic regression run: weighted_LR.m
3. For K-means clustering classification run: k_means.m
4. For SVM classification run: svm_train.m
5. For 3-layer neural network run: main.m
6. For 4-layer neural network run: main2.m

To run full flow, first download all the datasets mentioned above and follow these steps:
1. To segment data and extract features from all training datatsets run: run_LR_HSMM.m
This step will loop for validation, train-a, -b, -c, -d, -e, -f folders in ../ directory and save all extracted features from train directories into features_train.csv, labels for training examples are saved in classes_train.csv. It will save all extracted features from validation examples into features_val.csv, labels for validation examples are saved in classes_val.csv.
2. To run classification on extracted features follow classification steps.

Principal component analysis (PCA) can be run to reduce dimensionality of features. To run classification modes on transformed data:
1. Run pca.m. This will save features_train_pca.csv and features_val_pca.csv
2. All the classification models neet to set pca flag to 1 (or True). 

The following are files provided by Physionet:

    butterworth_high_pass_filter.m
    butterworth_low_pass_filter.m
    default_Springer_HSMM_options.m
    expand_qt.m
    getDWT.m
    get_duration_distributions.m
    getHeartRateSchmidt.m
    get_PSD_feature_Springer_HMM.m
    getSpringerPCGFeatures.m
    Hilbert_Envelope.m
    Homomorphic_Envelope_with_Hilbert.m
    labelPCGStates.m
    runSpringerSegmentationAlgorithm.m
    schmidt_spike_removal.m
    trainBandPiMatricesSpringer.m
    trainSpringerSegmentationAlgorithm.m
    viterbiDecodePCG_Springer.m
    viterbi_Springer.c
    viterbi_Springer.mexa64

The following are files provided by Physionet, but modified by us:

    extractFeaturesFromHsIntervals.m

The following are files created by us:

    (Neural Network Models: NN1, NN2)
    backward_prop2.m		  
    backward_prop.m
    cross_entropy.m
    sigmoid_func.m
    forward_prop2.m
    softmax_func.m
    forward_prop.m
    main.m
    main2.m
        
    (Logistic Regression)
    lr_model.m
    
    (Locally Weighted Logistic Regression)
    weighted_LR.m
    
    (K-means)
    k_means.m

    (Gaussian kernel SVM)
    svm_train.m
    svm_test.m

    (PCA for feature selection)
    pca.m
    
    (Data Segmentation and Feature Extraction)
    run_LR_HSMM.m
    normalise_signal.m
    
    (Error / Performance Analysis)
    error_analysis_LRHSMM.m
    evaluate_metric.m

The following are data files:
   
    (LR-HSMM trained parameters)
    B_matrix.mat
    pi_vector.mat
    total_obs_distribution.mat
    
    (LR-HSMM training dataset)
    example_data.mat
    
    (Extract features)
    features_lrhsmm.csv
    features_train.csv
    features_val.csv
    features_gt.csv
    
    (Data labels)
    classes_lrhsmm.csv
    classes_val.csv			  
    classes_train.csv		  
