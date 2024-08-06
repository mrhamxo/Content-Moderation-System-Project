# AI-Powered Content Moderation System

[Text Dataset link](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews/data)
[Image Dataset link](https://www.kaggle.com/datasets/kartikeybartwal/graphical-violence-and-safe-images-dataset)

## Introduction
  In the digital era, user-generated content on social media platforms, forums, and review sites can quickly become overwhelming and potentially harmful. Managing this content to ensure it meets community standards and regulations is crucial for maintaining a positive and safe environment. The AI-Powered Content Moderation System project aims to develop an advanced system that leverages deep learning techniques to automatically moderate user-generated content, detecting and filtering inappropriate material effectively and efficiently.
  This system will utilize state-of-the-art deep learning models to analyze both text and images, providing real-time filtering and reporting capabilities. Customizable moderation policies will allow for tailored content control based on platform-specific requirements and user preferences.

## Project Objectives
  Develop a deep learning-based system for moderating text and image content.
  Implement real-time filtering and reporting features to manage inappropriate content effectively.
  Design a customizable moderation framework to adapt to different platform policies and user needs.
  Ensure high accuracy and efficiency in content detection and moderation.

## Project Requirements
###	Text Moderation
-	Implement NLP model for detecting inappropriate Language, hate speech, Spam and other text-based issues.
-	Use Libraries such as SpaCy, NLTK or transformers from Hugging Face for text analysis and preprocessing.

###	Image Moderation
-	Develop deep learning models for detecting inappropriate or harmful images, including explicit content and graphic violence
- Use CNNs and pre-trained models from Tensorflow/Pytorch, such as inception or Resnet for image classification

## Text Moderation
  Text Moderation Using Transformers from Hugging Face
  Developing a Deep Learning Model Using Bert Transformer from Hugging Face. The aim is to develop a model which will classify text into Positive or Negative speech.
  
### Data Collection
  Data is collected from Kaggle Dataset of amazon reviews around 1.2 million reviews containing all nature of text, harmful speeches, spam etc.
  Extracted 70000 Views from that Dataset with the equal percentage of negative and positive reviews (balanced dataset). 
  Datasets contain negative review text with label as 1 and positive review text with label 0.

### Data Preprocessing
  Using BertTokenizer from Transformers from hugging face all the reviews are tokenized.
  Now the review text is transferred into input ids and attention mask.
  
### Model Development
  The final Deep learning model is developed using transformers pre trained BERT model. 
  The model is fine-tuned and perfectly trained.
  The model gave an accuracy of 93.6 % on Validation data.
  Got Precision, recall and F1 score 92.3%, 94.6%, 93.4% respectively.

## Image Moderation 
  Image Moderation Using CNN’S
  
###Data Collection
  Images Dataset is extracted from various sources including kaggle, Google. This allowed us to develop a dataset containing 7000 images, having equal proportion of safe and violent images (balanced dataset).
  
### Data Preprocessing
  Images are preprocessed and transformed thus creating a dataset of images.
  At first resized all the images in same width and height.
  Saved the images in numpy arrays, so we can easily provide input to the model.
  Scaled each Array to increase model performance.
  Applied augmentation to enhance image quality.
 
### Model Development
  A deep learning model is developed which classifies the image as violent or nonviolent.
  At first the dataset was split into training and validation sets.
  Loaded pre-trained model from keras application (Transfer Learning) MobilV2Net, freeze all the training layers.
  Added an augmentation layer after input layer of pre-trained model.
  Added a dropout layer after augmentation to avoid over fitting.
  Added a Flatten layer after dropout layer.
  Added a final output layer containing 1 neuron for binary classification and activation function used is sigmoid.
  Trained the model and got an accuracy of 92 percent on validation dataset.
  Got Precision, Recall and F1 Score in the range 90-95 Percent.


## Summary
Despite the division of tasks, team members consistently supported one another, ensuring the project was completed on time and with remarkable success.
As a result, two fine-tuned models were developed, achieving exceptional accuracy on the validation data. The contributions of each team member made the project significantly easier, allowing us to complete the entire project ahead of schedule.

