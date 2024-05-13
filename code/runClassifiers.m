% This script will train 3 different ML classifiers on fMRI data to try to
% classify each patient as either a healthy control, or a patient. There
% are several different patient populations in this sample. The classifiers
% used are: support vector machines (SVMs), linear discriminant analysis 
% (LDA), and decision trees. 

% clc; clear; close all;

%% load data
load('/Users/ajsimon/Documents/INP599/Final_project/data/connmats_235.mat');
load('/Users/ajsimon/Documents/INP599/Final_project/data/diagnostic_labels.mat');
load('/Users/ajsimon/Documents/INP599/Final_project/data/sublist_235.mat');

%% setup the data
% set parameters
nperms = 100;   % number of classification permutations 
nfolds = 10;    % number of cross validation folds

% turn the matrices into vectors
for m = 1:size(mats,3)
    thismat = squeeze(mats(:,:,m));
    D = diag(thismat);
    X(m,:) = [squareform((thismat-diag(D)).')];
    clear D thismat
end

clear mats

%% classify!
% loop through the number of psychiatric diagnostic labels we have
for l = 7:length(Diag_header)

    fprintf('Classifying %s \n',Diag_header{1,l});
    % setup the Y data for classification
    for s = 1:size(diags,1)
        if diags(s,1) == 1  % this is a HC
            Y{s,1} = Diag_header{1,1};
        elseif diags(s,l) == 1   % this person has the diagnosis we're trying to classify
            Y{s,1} = Diag_header{1,l};
        else  % this person has some other diagnosis
            Y{s,1} = 'Other diagnosis';
        end
    end

    % loop through the number of permutations we will be doing to evaluate classification performance
    for it = 1:nperms

        % partition data
        indices = cvpartition(length(Y),'k',nfolds);

        % shuffle Y 
        temp = randperm(numel(Y));
        Y_shuffled = reshape(Y(temp),size(Y));
        clear temp

        if it == 1
            fprintf('Permutation 1');
        else
            fprintf(' %d',it);
        end
        
        for f = 1:nfolds
            test.indx = indices.test(f);
            train.indx = indices.training(f);

            test.x = X(test.indx,:);
            train.x = X(train.indx,:);

            train.y = Y(indices.training(f),1);
            train.y_shuffled = Y_shuffled(indices.training(f),1);

            % train and test SVM 
            MdlSVM = fitcecoc(train.x,train.y);
            Y_pred_SVM(test.indx) = predict(MdlSVM,test.x);

            clear MdlSVM

            % train and test SVM on shuffled data
            MdlSVM = fitcecoc(train.x,train.y_shuffled);
            Y_pred_SVM_shuffled(test.indx) = predict(MdlSVM,test.x);

            clear MdlSVM
            
%             % train LDA model 
%             MdlLinear = fitcdiscr(train.x,train.y);
%             Y_pred_LDA(test.indx) = predict(MdlLinear,test.x);
% 
%             clear MdlLinear
% 
%             % train LDA model on shuffled data
%             MdlLinear = fitcdiscr(train.x,train.y_shuffled);
%             Y_pred_LDA_shuffled(test.indx) = predict(MdlLinear,test.x);

            clear MdlLinear
            
            % train decision tree classifier
            Mdltree = fitctree(train.x,train.y);
            Y_pred_tree(test.indx) = predict(Mdltree,test.x);

            clear Mdltree

            % train decision tree classifier on shuffled data
            Mdltree = fitctree(train.x,train.y_shuffled);
            Y_pred_tree_shuffled(test.indx) = predict(Mdltree,test.x);

            clear Mdltree

        end

        Y_pred_SVM = Y_pred_SVM';
        Y_pred_SVM_shuffled = Y_pred_SVM_shuffled';
%         Y_pred_LDA = Y_pred_LDA';
%         Y_pred_LDA_shuffled = Y_pred_LDA_shuffled';
        Y_pred_tree = Y_pred_tree';
        Y_pred_tree_shuffled = Y_pred_tree_shuffled';
        
        try
            conf_mat_SVM(:,:,it,l-1) = confusionmat(Y, Y_pred_SVM);
            acc_SVM(it,l-1) = sum(diag(conf_mat_SVM(:,:,it,l-1))) / sum(conf_mat_SVM(:,:,it,l-1),'all');
        catch
            conf_mat_SVM(:,:,it,l-1) = NaN;
            acc_SVM(it,l-1) = NaN;
        end

        try
            conf_mat_SVM_shuffled(:,:,it,l-1) = confusionmat(Y_shuffled, Y_pred_SVM_shuffled);
            acc_SVM_shuffled(it,l-1) = sum(diag(conf_mat_SVM_shuffled(:,:,it,l-1))) / sum(conf_mat_SVM_shuffled(:,:,it,l-1),'all');
        catch
            conf_mat_SVM_shuffled(:,:,it,l-1) = NaN;
            acc_SVM_shuffled(it,l-1) = NaN;
        end

%         conf_mat_LDA(:,:,it) = confusionmat(Y, Y_pred_LDA);
%         acc_LDA(it,1) = sum(diag(conf_mat_LDA)) / sum(conf_mat_LDA(:));
% 
%         conf_mat_LDA_shuffled(:,:,it) = confusionmat(Y_shuffled, Y_pred_LDA_shuffled);
%         acc_LDA(it,1) = sum(diag(conf_mat_LDA_shuffled)) / sum(conf_mat_LDA_shuffled(:));

        try
            conf_mat_tree(:,:,it,l-1) = confusionmat(Y, Y_pred_tree);
            acc_tree(it,l-1) = sum(diag(conf_mat_tree(:,:,it,l-1))) / sum(conf_mat_tree(:,:,it,l-1),'all');
        catch
            conf_mat_tree(:,:,it,l-1) = NaN;
            acc_tree(it,l-1) = NaN;
        end

        try
            conf_mat_tree_shuffled(:,:,it,l-1) = confusionmat(Y, Y_pred_tree_shuffled);
            acc_tree_shuffled(it,l-1) = sum(diag(conf_mat_tree_shuffled(:,:,it,l-1))) / sum(conf_mat_tree_shuffled(:,:,it,l-1),'all');
        catch
            conf_mat_tree_shuffled(:,:,it,l-1) = NaN;
            acc_tree_shuffled(it,l-1) = NaN;
        end

    end

    clear Y
end

%% visualize confusion matrices



