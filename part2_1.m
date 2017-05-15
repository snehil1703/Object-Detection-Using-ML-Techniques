% Author: Lijiang Guo
% Date: Mar 27th, 2017
% Project: CV A3 Part 2.1
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
num_img = 0;
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
%size(trainData(1).vecImg{1})
%class(trainData(1).vecImg{1})

%% 1. Eigenfood. Apply Principal Component Analysis (PCA) to 
% the training set of grayscale feature vectors extracted above. 
% (Note that CImg includes routines for Eigendecomposition.) 
% What do the top few Eigenvectors look like, when plotted as 
% images? How quickly do the Eigenvalues decrease? Using the top 
% k eigenvectors (where k is a number you?ll have to choose), 
% represent each image as a k-dimensional feature vector by 
% projecting the image into this lower-dimensional space. Then 
% use an SVM similar to the one above.

train_img = zeros(1600, num_img);
i = 1;
for k = 1:length(trainData)
    for f = 1:length(trainData(k).resizeImg)
    train_img(:,i) = trainData(k).vecImg{f};
    i = i+1;
    end
end

A = bsxfun(@minus, train_img, mean(train_img,2));
[U,S,V] = svd(A');

% Uncomment to plot eigenvalues
% figure
% subplot(1,2,1);
% plot(diag(S.^2));
% title('Eigenvalues of traing food');
% subplot(1,2,2);
% plot(log10(diag(S.^2)));
% title('Eigenvalues of traing food (log10)');
% saveas(gcf,'./graph/P2Q1_eigenvalues.jpg');
% close(gcf);

% Get eigenfood representation
k = 30;
U = U*S;
U_sel = U(:,1:k);

% Uncomment to plot first 9 eigenfoods
% figure
% subplot(3,3,1)
% imagesc(reshape(V(:,1),[new_img_size,new_img_size]))
% title('Eigenfood 1')
% subplot(3,3,2)
% imagesc(reshape(V(:,2),[new_img_size,new_img_size]))
% title('Eigenfood 2')
% subplot(3,3,3)
% imagesc(reshape(V(:,3),[new_img_size,new_img_size]))
% title('Eigenfood 3')
% subplot(3,3,4)
% imagesc(reshape(V(:,4),[new_img_size,new_img_size]))
% title('Eigenfood 4')
% subplot(3,3,5)
% imagesc(reshape(V(:,5),[new_img_size,new_img_size]))
% title('Eigenfood 5')
% subplot(3,3,6)
% imagesc(reshape(V(:,6),[new_img_size,new_img_size]))
% title('Eigenfood 6')
% subplot(3,3,7)
% imagesc(reshape(V(:,7),[new_img_size,new_img_size]))
% title('Eigenfood 7')
% subplot(3,3,8)
% imagesc(reshape(V(:,8),[new_img_size,new_img_size]))
% title('Eigenfood 8')
% subplot(3,3,9)
% imagesc(reshape(V(:,9),[new_img_size,new_img_size]))
% title('Eigenfood 9')
% saveas(gcf,'./graph/P2Q1_eigenfoods.jpg');
% close(gcf);

% save selected eigenvectors for SVM multi
if train == true
    eigenfood = fopen('./data/eigenfood_train.txt','w');
else
    eigenfood = fopen('./data/eigenfood_test.txt','w'); 
end
i = 1;
for a = 1:length(trainData)
    for b = 1:length(trainData(a).vecImg)
        example = num2str(trainData(a).class);
        for j = 1:k
            example = sprintf('%s %i:%f', example, j, U_sel(i,j));
        end
        fprintf(eigenfood,'%s\n',example);
        
        fprintf(1,'%s file %i processed.\n',trainData(a).name, b);
        i = i+1;
    end
end
fclose(eigenfood);

%% SVM
 % About multiclass SVML: https://nlp.stanford.edu/IR-book/html/htmledition/multiclass-svms-1.html
 % About SVMmulti software: https://www.cs.cornell.edu/people/tj/svm_light/svm_multiclass.html
 % Use SVM multiclass to train this
 % ./svm_multiclass_learn -c 1.0 ../../data/eigenfood_train.txt ../../data/eigenfood_model.txt
 % Testing
 % ./svm_multiclass_classify ../../data/eigenfood_test.txt ../../data/eigenfood_model.txt ../../data/eigenfood_pred.txt
% Reading model...done.
% Reading test examples... (250 examples) done.
% Classifying test examples...done
% Average loss on test set: 58.4000
% Zero/one-error on test set: 58.40% (104 correct, 146 incorrect, 250 total)
% k = 30
% Average loss on test set: 95.2000
% Zero/one-error on test set: 95.20% (12 correct, 238 incorrect, 250 total)
% k = 200
% Average loss on test set: 96.4000
% Zero/one-error on test set: 96.40% (9 correct, 241 incorrect, 250 total)

%% 2. Haar-like features. Similar to Viola and Jones, define 
% a set of many (probably thousands) of sums and differences 
% of rectangular regions at different positions and sizes 
% (e.g. randomly-chosen) in different configurations (see 
% Figure 1 of the Viola and Jones paper). Use Integral Images 
% to compute these efficiently. Instead of Adaboost or the 
% cascaded classifier used in Viola-Jones, simply compute each 
% feature for each image, put them in a feature vectopr, and 
% use an SVM to do the classification.


%% 3. Bags-of-words. Run SIFT on the training images (we?ve include 
% the Sift code again in your repo), and then cluster the 128-d 
% SIFT vectors into k visual words. Represent each training image 
% as a histogram over these k words, with a k-dimenional vector. 
% Learn an SVM similar to the one above. (You can either use an 
% existing implementation of k-means, with proper citations, of 
% course, although it is not hard to implement it yourself.)