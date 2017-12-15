# CS229_Project_HeartSound
Application of Machine Learning Techniques for Heart Sound Recording Classification By Vincent Lee and Anatoly Yakovlev

Dataset and LR-HSMM starter code were provided by Physionet: https://www.physionet.org/challenge/2016/

The following are files provided by Physionet:

    butterworth_high_pass_filter
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
    main3.m
    
    (Logistic Regression)
    lr_model.m
    
    (Locally Weighted Logistic Regression)
    weighted_LR.m
    
    (K-means)
    k_means_compress.m

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
