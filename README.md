# Project3_Group7

## FINAL PROJECT REQUIREMENTS:
* Identify a problem worth solving or analyzing.
* Find a dataset or datasets that are sufficiently large enough to effectively train a ML model or neural network with a high degree of accuracy to ensure that your results are reliable.
* Evaluate the trained model(s) using testing data. Include any calculations, metrics, or visualizations needed to evaluate the performance.

* You must use at least two of the following:
  - scikit-learn
  - Keras
  - TensorFlow
  - Hugging Face
  - spaCy or Natural Language Toolkit (NLTK)
  - LangChain
  - OpenAI

* You must use one additional library or technology NOT covered in class, such as:
  - Valence Aware Dictionary for Sentiment Reasoning (VADER)
  - Whisper (OpenAI’s automatic speech recognition system)
  - DALL·E (OpenAI’s text-to-image model)
  - Other OpenAI capabilities, including:
  - Text-to-speech
  - GPT-4 with vision (GPT-4V)
  - PyTorch

## PROJECT SUMMARY:
The dataset consists of 5,863 chest X-ray images organized into three folders: train, test, and val. Each folder has subfolders for two categories: Pneumonia and Normal.
The images were collected from pediatric patients (ages 1-5) at Guangzhou Women and Children’s Medical Center. They underwent quality control and were graded by two expert physicians, with a third expert verifying the evaluation set to ensure accuracy.

## RESULTS AND CONCLUSIONS
1. Libraries Imported: Numpy, TensorFlow, Keras, Matplotlib, Seaborn, Scikit-learn, and others.
2. Data Loading: Images from the 'train', 'test', and 'val' directories are loaded and labeled.
3. Data Preprocessing:
   * Images are resized to 256x256.
   * Data is normalized by dividing pixel values by 255 and then split the training data into Pneumonia and Normal subsets.
   * Data augmentation is applied separately to Pneumonia and Normal images to balance the dataset. Applied augmentation techniques (zoom, shift, flip) to the Pneumonia and Normal subsets using ImageDataGenerator. Expanded the datasets by generating augmented images and concatenating them with the original data.
4. Model Building:
  * Utilizes transfer learning with VGG16 as the base model.
  * Adds layers including Dropout and Dense for classification.
  * Freezes initial layers of VGG16 while fine-tuning the last few. Built a Sequential model with VGG16 as the base, followed by Flatten, Dropout, Dense layers, and an output layer with a sigmoid activation function.
5. Model Training: The model is compiled with Adam optimizer and binary crossentropy loss. It is trained over 6 epochs with accuracy metrics.

## OUTPUT 
  * Shapes of training, validation, and test sets are printed.
  * Training accuracy and validation scores are evaluated and printed.
  * The best validation accuracy achieved is displayed.

## MODEL DETAILS
  * Used VGG16 for transfer learning.
  * Freezed all but the last three layers of the VGG16 model.
  * Sequential model with additional layers.
  * Compiled with Adam optimizer and binary cross entropy 
  * Trained for 6 epochs.
  * Freezing the base model: Prevents the pre-trained weights from being updated, retaining the learned features from the original training dataset.
  * Adding custom layers: Allows you to adapt the model to your specific task and dataset.
  * Fine-tuning: Optionally, after initial training, unfreeze some layers and continue training with a lower learning rate to further adapt the pre-trained features to your dataset.
  * This approach leverages the powerful features learned by VGG16 on large datasets like ImageNet, while also allowing you to customize the model for your specific needs.
  * Test set for performance evaluation
  * Prediction
  * Confusion matrix

## DATASET PROPERTIES
![image](https://github.com/mvenegas011/Project3_Group7/assets/33967792/4b19937d-9732-4c56-875c-61d69cf59aa7)

## ACKNOWLEDGEMENTS 
Data: https://data.mendeley.com/datasets/rscbjbr9sj/2
License: CC BY 4.0
Citation: http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5
**More information can be found on https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data website.**
