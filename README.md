# Face Recognition Performance Comparison: LDA, SVM, and CNN

## Project Overview
This project aims to compare the performance of three different machine learning models—LDA, SVM, and CNN—for face recognition tasks. Using the Olivetti faces dataset, which consists of 400 64x64 images from 40 different subjects, the goal is to identify the subject of a given face image.

## Dataset
- **Olivetti Faces Dataset**: 400 images of size 64x64 pixels, featuring 40 different subjects.
- **Dataset Split**: The first 8 images per subject are used for training, and the last 2 images are used for testing. If a validation set is needed, it should be derived from the training data.

## Models to be Implemented
1. **LDA (Linear Discriminant Analysis)**
2. **SVM (Support Vector Machine)**
3. **CNN (Convolutional Neural Network)**
   - Suggested architecture: LeNet-5

## Deliverables
- **Final Code**: Fully functioning code (non-functioning code will receive 0 points).
- **Comparison Report**: Detailed comparison of model performances.

## Tasks
- **Train Models**:
  - LDA
  - SVM
  - CNN
    - **Plot Training Metrics**: Plot the loss and accuracy for each training epoch.
    - **Model Storage**: Save the model with the highest accuracy on the validation set.
- **Performance Comparison**:
  - **Average F-Score**
  - **Confusion Matrix**

## Evaluation Criteria
- **CNN Implementation**: 40 points
- **Plotting Loss and Accuracy**: 20 points
- **Storing Best Model**: 10 points
- **Average F-Score Calculation**: 15 points
- **Confusion Matrix Analysis**: 15 points

## Files Included
- Initial code with problem specification
- Report template
- Dataset: `faces.png`

For more details and the initial version of the code, check out the provided link: [HERE].
