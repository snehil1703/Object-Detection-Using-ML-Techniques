% Author: Lijiang Guo
% Date: Mar 27th, 2017
% Project: CV A3 Part 1
%% Parameter set up
% For preparing training data, set train = true
% For preparing testing data, set train = false
train = true; color = false ; new_img_size = 40;

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
num_img = 0
for k = 1: length(trainData)
    fprintf(1,'%i: %s has %i images.\n', trainData(k).class, ...
        trainData(k).name, length(trainData(k).image));
    num_img = num_img + length(trainData(k).image);
end

%% Resize images to 40 by 40 pixels.
if color == true
    % color
    trainData = resize_image(trainData,'color',new_img_size,new_img_size);
else 
    % grey
    trainData = resize_image(trainData,'grey',new_img_size,new_img_size);
end

% figure
% subplot(1,2,1)
% imshow(trainData(10).resizeImg{10})
% subplot(1,2,2)
% imshow(trainData(10).image{10})
% pause % press enter to continue
% close all

%% Reformat images into vectors
for j = 1:length(trainData)
    for k = 1:length(trainData(j).image)
        trainData(j).vecImg{k} = trainData(j).resizeImg{k}(:);
    end
end
size(trainData(1).vecImg{1})
class(trainData(1).vecImg{1})

% Save resized images for eigenfood
%save('./grey_testData.mat','trainData');

% save resized images for SVM multi
if train == true
    if color == true
        file_img_vec = fopen('./data/color_train.txt','w');
    else 
        file_img_vec = fopen('./data/grey_train.txt','w');
    end
else
    if color == true
        file_img_vec = fopen('./data/color_test.txt','w');
    else 
        file_img_vec = fopen('./data/grey_test.txt','w');
    end
end

for k = 1:length(trainData)
    for n = 1:length(trainData(k).vecImg)
        example = num2str(trainData(k).class);
        x = trainData(k).vecImg(n);
        x= x{:};
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
 % Use SVM multiclass to train this
 % ./svm_multiclass_learn -c 1.0 ../../data/grey_train.txt ../../data/grey_model.txt
 % ./svm_multiclass_learn -c 1.0 ../../data/color_train.txt ../../data/color_model.txt
 % Testing
 % ./svm_multiclass_classify ../../data/grey_test.txt ../../data/grey_model.txt ../../data/grey_pred.txt
 % ./svm_multiclass_classify ../../data/color_test.txt ../../data/color_model.txt ../../data/color_pred.txt
% Results:
% grey:
% Average loss on test set: 92.0000
% Zero/one-error on test set: 92.00% (20 correct, 230 incorrect, 250 total)
% color:
% Average loss on test set: 84.8000
% Zero/one-error on test set: 84.80% (38 correct, 212 incorrect, 250 total)

 
