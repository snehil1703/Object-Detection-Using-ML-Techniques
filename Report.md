#Implementation Details
 - We have used overfeat binary for feature extraction

 - Train
 ..1. If extract_features_overfeat = 1 in Deep.h file then overfeat library is used to extract features again, else SVM uses already extracted features. By default it is set to 0 as it takes too long to extract the features.
  ..2. Each Image is resized to 231x231 (minimum size of Image required by overfeat) and passed to overfeat binary
  ..3. The extracted features are then written to train_features file in the format expected by svm_multiclass library
  ..4. svm_multiclass_learn is used to train the model_file
      - Kernel : RBF
      - Gamma : 0.0625
      - Cost : 3.2


 - Test
 ..1. Resize the image to 231x231 and extract the features.
 ..2. Use svm_multiclass_classify to predict the class of the image

#Comparison
  - Algo = Deep takes much longer than any other Algo to train as well as test, but gives the best accuracy.
