% Author: Lijiang Guo
% Date: Mar 27th, 2017
% Project: CV A3 Part 2
%%
% We use SIFT Matlab package created by Andrea Vedaldi.
% http://vision.ucla.edu/~vedaldi/code/sift.html
addpath './sift';
%% Parameter set up
% For preparing training data, set train = true
% For preparing testing data, set train = false
train = true; color = false ; new_img_size = 300;

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

%% Prepare window templates for Harr features
% We will only create such templates: a rectangle, divide into four
% qundrants, each quadrant is either all 1 or all -1. 
template_model = [1 -1 1 -1; % up-left, low_left, up-right, low-right
    -1 1 -1 1;
    1 1 -1 -1;
    -1 -1 1 1;
    1 -1 -1 1;
    -1 1 1 -1]
% Uncomment to plot template patterns
% figure;
% subplot(3,2,1)
% imagesc(reshape(template_model(1,:),[2,2]));
% subplot(3,2,2)
% imagesc(reshape(template_model(2,:),[2,2]));
% subplot(3,2,3)
% imagesc(reshape(template_model(3,:),[2,2]));
% subplot(3,2,4)
% imagesc(reshape(template_model(4,:),[2,2]));
% subplot(3,2,5)
% imagesc(reshape(template_model(5,:),[2,2]));
% subplot(3,2,6)
% imagesc(reshape(template_model(6,:),[2,2]));
% saveas(gcf,'./graph/P2Q2_harr.jpg');
% close(gcf);

template_size = [floor(new_img_size/2), floor(new_img_size/2); % height, width
    floor(new_img_size/4), floor(new_img_size/4);
    floor(new_img_size/8), floor(new_img_size/8);
    floor(new_img_size/12), floor(new_img_size/12);
    floor(new_img_size/16), floor(new_img_size/16)]

num_template = size(template_model,1)*size(template_size,1);
template = cell(num_template, 1);
t_i = 1; % template index
for i = 1:size(template_size,1)
    h_mid = floor(template_size(i,1)/2);
    w_mid = floor(template_size(i,2)/2);
    
    for j = 1:size(template_model,1)
        template{t_i} = ones(template_size(i,:));
        for k = 1:4
            switch k
                case 1
                    template{t_i}(1:h_mid,1:w_mid) = template{t_i}(1:h_mid,1:w_mid)*template_model(i,k);
                case 2
                    template{t_i}(h_mid+1:end,1:w_mid) = template{t_i}(h_mid+1:end,1:w_mid)*template_model(i,k);                    
                case 3
                    template{t_i}(1:h_mid,w_mid+1:end) = template{t_i}(1:h_mid,w_mid+1:end)*template_model(i,k);                    
                case 4
                    template{t_i}(h_mid+1:end,w_mid+1:end) = template{t_i}(h_mid+1:end,w_mid+1:end)*template_model(i,k);                    
            end
        end
        t_i = t_i + 1;
    end
end
%template{num_template}
%% Extract Haar feature for each image
for i_food = 1:length(trainData)
    for i_img = 1:length(trainData(k).image)        
        % Apply templates to get all features
        harr = zeros(num_template,1);
        for i_t = 1:num_template
            %temp_h = template_size(mod(i_t-1,size(template_size,1))+1,1);
            %temp_w = template_size(mod(i_t-1,size(template_size,1))+1,2);
            i_size = floor((i_t-1)/size(template_model,1))+1;
            temp_h = template_size(i_size,1);
            temp_w = template_size(i_size,2);
            position_r = randi(new_img_size - temp_h + 1);
            position_c = randi(new_img_size - temp_w + 1);
            
            harr(i_t) = sum(sum(trainData(i_food).image{i_img}(position_r:position_r+temp_h-1, ...
                position_c:position_c+temp_w-1) .* template{i_t}));
            
            %example = sprintf('%s %i:%f', example, i, x(i));
            
        end
        trainData(i_food).harr{i_img} = harr;        
        fprintf(1,'Extracted Harr features for %s file %i.\n',trainData(i_food).name, i_img);
    end
end
%trainData(1).harr{1}

%% save BOG features for SVM multi
if train == true
    file_img_vec = fopen('./data/harr_train.txt','w');
else
    file_img_vec = fopen('./data/harr_test.txt','w');
end

for k = 1:length(trainData)
    for n = 1:length(trainData(k).harr)
        example = num2str(trainData(k).class);
        x = trainData(k).harr{n};
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
 % Use SVM multiclass to train this
 % ./svm_multiclass_learn -c 1.0 ../../data/harr_train.txt ../../data/harr_model.txt
 % Testing
 % ./svm_multiclass_classify ../../data/harr_test.txt ../../data/harr_model.txt ../../data/harr_pred.txt
% Results:
% images resizes to 300 by 300, 30 harr features (5 sizes, 6 models)
% Average loss on test set: 94.0000
% Zero/one-error on test set: 94.00% (15 correct, 235 incorrect, 250 total)
