# Sound Classification with ESC-50

## Project Overview

This project focuses on sound classification using the ESC-50 dataset, a widely used dataset for environmental sound classification tasks. The primary goal of the project was to develop a machine learning model capable of accurately identifying different sound categories from the dataset. By leveraging modern techniques, the project achieved an **accuracy of approximately 98% on the training set** and **96% on the test set**.

---

## About the ESC-50 Dataset

The ESC-50 dataset consists of 2,000 labeled environmental audio recordings spanning 50 classes. These classes are grouped into 5 broad categories:
- **Animals**
- **Natural soundscapes and water sounds**
- **Human, non-speech sounds**
- **Interior/domestic sounds**
- **Exterior/urban noises**

Each sound clip is 5 seconds long and stored as a WAV file with consistent sampling.

More about the dataset: [ESC-50 Dataset on GitHub](https://github.com/karoldvl/ESC-50)

---

## Technical Highlights

### Data Preprocessing
- Converted raw audio data to Mel-spectrograms for feature extraction, as this representation captures frequency and time domain information effectively.
- Applied data augmentation techniques, such as pitch shifting and time stretching, to improve model generalization.

### Model Architecture
- **Neural Network:** A custom Convolutional Neural Network (CNN) was designed to handle the Mel-spectrogram input.
- **Layers:** The architecture includes multiple convolutional and pooling layers, followed by dense layers for classification.
- **Regularization:** Techniques such as dropout were used to prevent overfitting.

### Training Details
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Metrics: Accuracy
- Split: 80% training, 20% testing

### Achievements
- **Training Accuracy:** ~98%
- **Testing Accuracy:** ~96%
- The model demonstrates robust performance across all 50 sound classes.

---

## Results

The classification model was evaluated using confusion matrices and accuracy metrics. The high test accuracy indicates the effectiveness of the model and its ability to generalize across diverse environmental sound categories.

---
