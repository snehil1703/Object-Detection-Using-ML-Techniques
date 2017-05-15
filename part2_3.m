% Author: Lijiang Guo
% Date: Mar 27th, 2017
% Project: CV A3 Part 1
%%
% We use SIFT Matlab package created by Andrea Vedaldi.
% http://vision.ucla.edu/~vedaldi/code/sift.html
addpath './sift';
%% Parameter set up
% For preparing training data, set train = true
% For preparing testing data, set train = false
train = true; color = false ; new_img_size = 200; num_cluster = 100;

%% Import images
foodDir='./data/fooddata2';

if train == true
    % training
    trainDir = strcat(foodDir,'/train');
else 
    % testing
    trainDir = strcat(foodDir,'/test');
end

trainData = get_image(trainDir);
num_img = 0;
for k = 1: length(trainData)
    fprintf(1,'%i: %s has %i images.\n', trainData(k).class, ...
        trainData(k).name, length(trainData(k).image));
    num_img = num_img + length(trainData(k).image);
end
fprintf('*********************\n');
fprintf('Imported %i images.\n', num_img);
fprintf('*********************\n');

%% Resize image (original image has too many sift features)
if color == true
    % color
    trainData = resize_image(trainData,'color',new_img_size,new_img_size);
else 
    % grey
    trainData = resize_image(trainData,'grey',new_img_size,new_img_size);
end
for k = 1:length(trainData)
    trainData(k).image = trainData(k).resizeImg;
end
trainData(1).image

%% Extract SIFT feature for each image
num_sift_all = 0;
for j = 1:length(trainData)
    for k = 1:length(trainData(j).image)        
        [~, X_desc] = sift(trainData(j).image{k});
%        [~, X_desc] = sift(rgb2gray(trainData(j).image{k}));
        trainData(j).desc{k} = X_desc;
        trainData(j).num_desc{k} = size(X_desc,2);
        num_sift_all = num_sift_all + size(X_desc,2);
 
        fprintf(1,'Extracted %i SIFT features for %s image %i.\n',size(X_desc,2),...
            trainData(j).name, k);
        drawnow;
    end
end

%% Prepare training SIFT data for K-means clustering
if train == true
    sift_all = zeros(128, num_sift_all);
    i = 1;
    for j = 1:length(trainData)
        for k = 1:length(trainData(j).image)
            sift_all(:,i:i+trainData(j).num_desc{k}-1) = trainData(j).desc{k};
            i = i+trainData(j).num_desc{k};
            fprintf(1,'Merged %i SIFT features for %s image %i.\n',trainData(j).num_desc{k},...
                trainData(j).name, k);
            drawnow;
        end
    end
    
    % Use K means clustering to find k visual words
    fprintf('*********************\n');
    fprintf(1,'Begin clustering on SIFT features.\n');
    fprintf('*********************\n');

    [idx,C] = kmeans(sift_all', num_cluster);
    i = 1;
    for j = 1:length(trainData)
        for k = 1:length(trainData(j).image)
            trainData(j).bow{k} = get_bow(idx(i:i+trainData(j).num_desc{k}-1),num_cluster);
            i = i + trainData(j).num_desc{k};
            fprintf(1,'Processed BOG features for %s image %i.\n',...
                trainData(j).name, k);
            drawnow;
        end
    end
    % Save resized images for eigenfood
    save('./data/bow_centroids.mat','C');
end

%% Get BOG features for test data
if train == false
    for j = 1:length(trainData)
        for k = 1:length(trainData(j).image)
            trainData(j).bow{k} = get_bow2(trainData(j).desc{k},C');
            fprintf(1,'Processed BOG features for %s image %i.\n',...
                trainData(j).name, k);
            drawnow;
        end
    end
end

%% save BOG features for SVM multi
if train == true
    file_img_vec = fopen('./data/bow_train.txt','w');
else
    file_img_vec = fopen('./data/bow_test.txt','w');
end

for k = 1:length(trainData)
    for n = 1:length(trainData(k).desc)
        example = num2str(trainData(k).class);
        x = trainData(k).bow{n};
        %x= x{:};
        for i = 1:length(x)
            example = sprintf('%s %i:%f', example, i, x(i));
        end
        fprintf(file_img_vec,'%s\n',example);
        
        fprintf(1,'%s file %i processed.\n',trainData(k).name, n);
    end
end
fclose(file_img_vec);

%% SVM
 % About multiclass SVML: https://nlp.stanford.edu/IR-book/html/htmledition/multiclass-svms-1.html
 % About SVMmulti software: https://www.cs.cornell.edu/people/tj/svm_light/svm_multiclass.html
 %
 % Use SVM multiclass to train this
 % ./svm_multiclass_learn -c 1.0 ../../data/bow_train.txt ../../data/bow_model.txt
 % Testing
 % ./svm_multiclass_classify ../../data/bow_test.txt ../../data/bow_model.txt ../../data/bow_pred.txt
 %
 % Results:
% images resizes to 200 by 200, 100 clusters, 543497 sift features in
% training set.
% Runtime (without IO) in cpu-seconds: 0.00
% Average loss on test set: 71.6000
% Zero/one-error on test set: 71.60% (71 correct, 179 incorrect, 250 total)
% images resizes to 300 by 300, 100 clusters, 233496 sift features in
% training set.
% Average loss on test set: 67.2000
% Zero/one-error on test set: 67.20% (82 correct, 168 incorrect, 250 total)
