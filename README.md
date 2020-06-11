# Face_Mask_Detection
Face mask detection using Inception Network
Mask Classifier

Covvid-19 is here to stay for some time. While a lot of esteemed scientists are developing vaccine to fight the virus, we need to ensure that proper guuidelines are being adapted at various places to prevent the spread of virus. Keeping this intenton in mind, this face mask detector algorithm has been developed.

dataset credits : @prajnasb @Prajna Bhandary

Method adapted:

1. Segregate directories into train and vaidation with sub-directories as "yes" or "no"
2. Visualize that data has been split correctly
3. Train model using pretrained Inception_v3 Model from Tensorflow Hub or Baseline model consisting of CNN and Max Pool Layers.
4. Used the trained model to predict face mask detection usiing your Webcam.

Experimental Procedure:
  * Utils scripts data_segment.py is imported in train_mask
  * Baseline Model script baseline_cnn.py is imported in train_mask
  * Inception Model script inception_cnn is imported in train_mask
  * Run train_mask.ipynb


Accuracy:
Train accuracy: 99.6%
Test accuracy: 99.4%


Applications:

Face mask detector can be used at various public places to ensure that people are practicing proper guidelines.
Some of the public places include:

a. Public Transportation like Railways, Metros, buses etc.
b. Schools/educational institutions
c. Corporate Offices

Scope for Improvement:

a. Computation power can be reduced so as to make it more efficient with embedded devices with low computation power
b. Dataset is artificially created and has masks only of a particular type, more real/artifical data can be added with different masks      style and types(N-95 masks or colored masks etc.)
c. In this dataset, samples of images with mask and without mask is of sample, this creates a identity bias which will definitely cause    a problem in real world application, more data of people with no mask should be added to remove that bias.

