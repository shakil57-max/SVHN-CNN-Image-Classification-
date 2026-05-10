#SVHN Digit Classification using CNN

This project implements a **Convolutional Neural Network (CNN)** to classify digits from the SVHN (Street View House Numbers) dataset.

#Project Overview
- **Dataset**: SVHN (Stanford University)
- **Training Samples**: 73,257
- **Test Samples**: 26,032
- **Classes**: 10 (Digits 0-9)
- **Framework**: TensorFlow & Keras

#Model Architecture
- 3 Convolutional Layers with MaxPooling
- Flatten Layer
- Dense Layer with Dropout
- Softmax Output (10 classes)

#Features
- Proper data loading and preprocessing
- Data normalization (0-1)
- Loss & Accuracy visualization
- Sample predictions with visualization
- Reproducible results

#Technologies Used
- Python
- TensorFlow / Keras
- Matplotlib
- NumPy
- SciPy

#How to Run
```bash
pip install -r requirements.txt
python main.py 
