
### Problem Statement: ALZHEIMER'S DISEASE CLASSIFICATION USING BRAIN MRI IMAGES
References: 
1. Improving Alzheimer’s Disease Classification in Brain MRI Images Using a Neural Network Model Enhanced with PCA and SWLDA
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10530944/#:~:text=A%20deep%2Dlearning%2Dbased%20model,mild%20dementia%2C%20and%20moderate%20dementia.
2. https://www.kaggle.com/datasets/madhucharan/alzheimersdisease5classdatasetadni
3. https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset

### Background
This project is an image classification task using MRI brain images to detect the presence of Alzheimer's disease. Data was obtained from Kaggle (Reference 2, 3).I chose to use the dataset from Reference 3  because there were more images(6400 vs 1100). The dataset chosen classified the images into 4 categories showing different progressions of Alzheimers as well as a cognitively normal brain (no Alzheimers). I combined the 3 varying cases of Alzheimers into a 'Dementia' category and the normal brain into a 'Non-Dementia' category within the final notebook. This helped the modelling as the dataset was very imbalanced: Mild Dementia(896), Moderate Dementia(64), VeryMildDemented(2240), NonDemented(3200). So I modified this to a binary classification problem. However I attempted the multi-classification problem once the model had been built and tuned for the binary problem.
### Description 

There are several model runs showing the various stages of progress with training of the model. I implemented a Keras deep learning model, XCeption, an advanced, high performance CNN model - in particular the Xception model from its ImageNet API.There are 3 jupyter model notebooks:
1. nn_alz_models_part1 - 2 model runs for binary classification and the initial multi-classification run before the focus on binary classification
2. nn_alz_final - model tuning/exploration to improve accuracy and includes the final model for the binary classification problem
3. nn_alz_multiclassification  - revisiting the multi-classification problem now that the model has been optimised.


DataProcessing/Cleaning included resizing/rescaling, normalization and image augmentation. I used transfer learning to configure the modelling to this particular problem. This involved setting up the model configuration as below:
Firstly, removed the top layer to allow addition of custom layers for binary classification
Used input_shape = (224, 224, 3) 
base_model = Xception(weights='imagenet', include_top=False,    input_shape=input_shape)
   	base_model.trainable = True
   	 for layer in base_model.layers[:-8]:
        		layer.trainable = False
This allows leveraging of both the pre-learned features and capacity to adapt to new features

Secondly, final model had the following components: (refer to the final cell in nn_alz_final.ipynb)
Model Components:
1. Base Model: Xception nodel, a pre-trained convolutional neural network. It is configured with include_top=False to leverage learned features and allows adaption to this specific problem. 
2. GlobalAveragePooling2D:This layer replaces the fully connected layers by computing the mean of each feature map from the previous layer. It helps to reduce the model's parameter count and controls overfitting, making the model more robust. Reduce spatial dimensions of the output from base_model.
3. Dense Layer (128 units, ReLU activation) with a kernel regularizer L2(0.01) applies a penalty on layer parameters during optimization, which can help prevent overfitting by discouraging large weights.
4. BatchNormalization:This helps stabilize the learning process and reduces the number of epochs needed to train the model.
5. Dropout (0.5): The dropout rate of 0.5 implies that half of the input units to this layer will be zeroed at each step during training, which significantly helps in preventing overfitting.

The following techniques were used: callback using earlystopping to stop training if validation loss did not imporve and ReduceLROnPlateau which dynamically adjusts the learning rate. I also addressed class imbalance using comute_class_weights to manage underrepresented classes. However for the final model, the augmentation which was enhanced in the earlier models were stripped back as it was likely that the agumentation was too aggressive and I slowed down learning even further. This may have been a significant factor in the accuracy scores jumping from 78% to 92% for the binary classification problem. 

# Executive Summary
A 92% accuracy rate was achieved for the binary classification problem - detecting the presence of alzheimers or a cognitively normal MRI brain scan correctly 95% of the time. To achieve this, there were several parameters and techniques tested, two different optimisers tested and considerable model tuning. This model was then applied to the multiclass problem to see how well the model performed with 4 classes showing varying degress of progression of Alzheimer's and this was equally successful (accuracy of 93%).

