 function EVAL = Evaluate(ACTUAL,PREDICTED)
% =========================================================================
% DESCRIPTION
% This function evaluates the performance of a classification model by 
% calculating the common performance measures: Accuracy, Sensitivity, 
% Specificity, Precision, Recall, F-Measure.

% INPUT
% [ACTUAL]      Column vector with actual class labels of the training
%               examples (binary)
%               Group1 = 0 | Group2 = 1
% [PREDICTED]   Column vector with predicted class labels by the
%               classification model(binary)
%               Group1 = 0 | Group2 = 1

% OUTPUT
% [EVAL]         Row vector with all the performance measures
% =========================================================================

idx = (ACTUAL()==1);
p = length(ACTUAL(idx));
n = length(ACTUAL(~idx));
N = p+n;
tp = sum(ACTUAL(idx)==PREDICTED(idx));
tn = sum(ACTUAL(~idx)==PREDICTED(~idx));
fp = n-tn;
fn = p-tp;
tp_rate = tp/p;
tn_rate = tn/n;
accuracy = (tp+tn)/N;
sensitivity = tp_rate;
specificity = tn_rate;
precision = tp/(tp+fp);
recall = sensitivity;
f_measure = 2*((precision*recall)/(precision + recall));

EVAL = [accuracy sensitivity specificity precision recall f_measure 0]; % the last zero is
% to put the std after the crossvalidation
 end