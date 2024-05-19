# Project3_Group7
**Final Project Requirements:
Identify a problem worth solving or analyzing.
Find a dataset or datasets that are sufficiently large enough to effectively train a ML model or neural network with a high degree of accuracy to ensure that your results are reliable.
Evaluate the trained model(s) using testing data. Include any calculations, metrics, or visualizations needed to evaluate the performance.

You must use at least two of the following:
- scikit-learn
- Keras
- TensorFlow
- Hugging Face
- spaCy or Natural Language Toolkit (NLTK)
- LangChain
- OpenAI

You must use one additional library or technology NOT covered in class, such as:
- Valence Aware Dictionary for Sentiment Reasoning (VADER)
- Whisper (OpenAI’s automatic speech recognition system)
- DALL·E (OpenAI’s text-to-image model)
- Other OpenAI capabilities, including:
- Text-to-speech
- GPT-4 with vision (GPT-4V)
- PyTorch

Project Summary
The project involves analyzing chest X-ray images to distinguish between Pneumonia and Normal cases. The dataset consists of 5,863 images categorized into training, testing, and validation sets, sourced from pediatric patients aged one to five years. Quality control was ensured by expert physicians.

Code Summary
Libraries Imported: Numpy, TensorFlow, Keras, Matplotlib, Seaborn, Scikit-learn, and others.
Data Loading: Images from the 'train', 'test', and 'val' directories are loaded and labeled.
Data Preprocessing:
Images are resized to 256x256.
Data is normalized by dividing pixel values by 255.
Data augmentation is applied separately to Pneumonia and Normal images to balance the dataset.
Model Building:
Utilizes transfer learning with VGG16 as the base model.
Adds layers including Dropout and Dense for classification.
Freezes initial layers of VGG16 while fine-tuning the last few.
Model Training: The model is compiled with Adam optimizer and binary crossentropy loss. It is trained over 6 epochs with accuracy metrics.
Output
Shapes of training, validation, and test sets are printed.
Training accuracy and validation scores are evaluated and printed.
The best validation accuracy achieved is displayed.
