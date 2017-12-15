function evaluate_metric(prediction, actual)
TP = 0;
FN = 0;
FP = 0;
TN = 0;

for i=1:length(prediction)
    % prediction matches observation
    if((prediction(i) == 1) && (actual(i) == 1)) 
        TP = TP + 1;
    elseif ((prediction(i) == -1) && (actual(i) == -1))
        TN = TN + 1;
    % predict abnormal, but observe normal => FP    
    elseif ((prediction(i) == 1) && (actual(i) == -1))
        FP = FP + 1;
    % predict normal, but observe abnormal => FN    
    elseif ((prediction(i) == -1) && (actual(i) == 1))
        FN = FN + 1;
    end
end   

% Performance Metric
disp('============= Performance Metric ============');
Num_abnormal = sum(actual == 1);
Num_normal = sum(actual == -1);
fprintf('Observations: Abnormal Count: %d , Normal Count: %d \n', Num_abnormal, Num_normal);
fprintf('TP %d, TN %d, FN %d, FP %d \n', TP, TN, FN, FP);
Se = (TP./(TP + FN));
P_plus = (TP./(TP + FP));
Acc = (TP + TN)./(TP + FP + FN + TN);
F1 = (2.*Se.*P_plus)./(Se + P_plus);
fprintf('Se %f, P_plus %f, Acc %f, F1 %f \n', Se, P_plus, Acc, F1);

end