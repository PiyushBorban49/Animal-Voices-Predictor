# 🐾 Animal Sound Classification using Deep Learning

A comprehensive machine learning project that classifies animal sounds using audio processing techniques and deep neural networks. This system can identify 13 different animal species from their audio recordings using MFCC (Mel-Frequency Cepstral Coefficients) features and a deep learning model.

## 🎯 Project Overview

This project implements an end-to-end pipeline for animal sound classification:
- **Audio Processing**: Converts raw audio files into MFCC features
- **Feature Engineering**: Standardizes and normalizes audio features
- **Deep Learning**: Uses a multi-layer neural network for classification
- **Model Optimization**: Employs callbacks for better training performance

## 🦁 Supported Animals

The model can classify sounds from the following 13 animals:
- **Bear** 🐻
- **Cat** 🐱
- **Chicken** 🐔
- **Cow** 🐄
- **Dog** 🐕
- **Dolphin** 🐬
- **Donkey** 🫏
- **Elephant** 🐘
- **Frog** 🐸
- **Horse** 🐎
- **Lion** 🦁
- **Monkey** 🐒
- **Sheep** 🐑

## 📁 Project Structure

```
Animal-Sound-Classification/
├── Animal-Soundprepros/           # Audio dataset directory
│   ├── Bear/
│   │   ├── Bear_1.wav
│   │   ├── Bear_2.wav
│   │   └── ...
│   ├── Cat/
│   │   ├── Cat_1.wav
│   │   └── ...
│   └── ...
├── others/
│   └── testWorld.py              # External module
├── animal_classifier.py          # Main classification script
└── README.md                     # This file
```

## 🔧 Installation & Requirements

### Prerequisites
```bash
pip install librosa
pip install tensorflow
pip install scikit-learn
pip install xgboost
pip install matplotlib
pip install numpy
pip install IPython
```

### System Requirements
- Python 3.7+
- TensorFlow 2.x
- At least 4GB RAM (recommended 8GB+)
- Audio files in WAV format

## 🎵 How It Works

### 1. **Audio Feature Extraction**
```python
# Load audio file (10 seconds duration)
y, sr = librosa.load(audio_file, duration=10.0)

# Extract MFCC features (20 coefficients)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
```

**MFCC Features**: Mel-Frequency Cepstral Coefficients capture the most important characteristics of audio signals, making them ideal for sound classification.

### 2. **Data Preprocessing**
- **Padding/Truncation**: All audio features are normalized to 100 time frames
- **Flattening**: 2D MFCC arrays are flattened into 1D feature vectors
- **Standardization**: Features are scaled using StandardScaler for better convergence

### 3. **Neural Network Architecture**

```
Input Layer (2000 features)
    ↓
Dense Layer (512 neurons) + ReLU + L2 Regularization
    ↓
Batch Normalization + Dropout (0.5)
    ↓
Dense Layer (256 neurons) + ReLU + L2 Regularization
    ↓
Batch Normalization + Dropout (0.5)
    ↓
Dense Layer (128 neurons) + ReLU + L2 Regularization
    ↓
Batch Normalization + Dropout (0.5)
    ↓
Dense Layer (64 neurons) + ReLU + L2 Regularization
    ↓
Batch Normalization + Dropout (0.5)
    ↓
Output Layer (13 classes) + Softmax
```

### 4. **Training Configuration**
- **Loss Function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Validation Split**: 20% of data

## 🚀 Usage

### Running the Classification
```bash
python animal_classifier.py
```

### Expected Dataset Structure
Place your audio files in the following structure:
```
Animal-Soundprepros/
├── Bear/
│   ├── Bear_1.wav
│   ├── Bear_2.wav
│   └── ... (up to Bear_49.wav)
├── Cat/
│   ├── Cat_1.wav
│   └── ...
└── [Other Animals]/
```

### File Naming Convention
- Files should be named: `{AnimalName}_{number}.wav`
- Numbers range from 1 to 49 for each animal
- Total expected files: 13 animals × 49 files = 637 audio files

## 📊 Model Performance Features

### Regularization Techniques
- **L2 Regularization**: Prevents overfitting by penalizing large weights
- **Dropout**: Randomly sets 50% of neurons to zero during training
- **Batch Normalization**: Normalizes inputs to each layer

### Training Optimization
- **Early Stopping**: Monitors validation loss and stops training when no improvement
- **Patience**: Waits 10 epochs before stopping
- **Best Weights Restoration**: Automatically restores the best model weights

## 📈 Understanding the Output

### Training Visualization
The script generates a loss curve showing:
- **Training Loss** (blue line): How well the model fits training data
- **Validation Loss** (orange line): How well the model generalizes

### Ideal Training Patterns
- Both losses should decrease over time
- Validation loss should not increase significantly (overfitting indicator)
- Convergence indicates good model performance

## 🔄 Key Configuration Parameters

### Audio Processing
```python
n_mfcc = 20        # Number of MFCC coefficients
max_len = 100      # Maximum time frames
duration = 10.0    # Audio duration in seconds
```

### Model Architecture
```python
# Layer sizes
layers = [512, 256, 128, 64, 13]

# Regularization
l2_reg = 0.001     # L2 regularization strength
dropout = 0.5      # Dropout rate
```

### Training Parameters
```python
batch_size = 32    # Training batch size
epochs = 50        # Maximum training epochs
patience = 10      # Early stopping patience
test_size = 0.2    # Validation split ratio
```

## 🛠️ Customization Options

### Adding New Animals
1. Add the animal name to the `animals` list
2. Create a corresponding folder in the dataset directory
3. Add audio files following the naming convention

### Modifying Architecture
- Adjust layer sizes in the `Sequential` model
- Modify regularization parameters
- Change activation functions or add new layers

### Tuning Parameters
- Experiment with different MFCC coefficients (`n_mfcc`)
- Adjust audio duration for longer/shorter clips
- Modify dropout rates and L2 regularization strength

## 🔍 Troubleshooting

### Common Issues

**File Not Found Errors**
- Ensure audio files exist in the specified directory
- Check file naming convention matches exactly
- Verify folder structure is correct

**Memory Issues**
- Reduce batch size
- Decrease the number of MFCC coefficients
- Use shorter audio durations

**Poor Performance**
- Increase training data
- Adjust model architecture
- Experiment with different preprocessing techniques

## 📝 Technical Details

### Feature Vector Size
- MFCC shape: (20, 100) = 20 coefficients × 100 time frames
- Flattened size: 2000 features per audio file
- Total parameters: ~2.4 million (depending on exact architecture)

### Memory Requirements
- Training data: ~637 samples × 2000 features × 4 bytes ≈ 5MB
- Model size: ~10MB (depending on architecture)
- Peak memory usage: ~500MB during training

## 🎓 Learning Resources

### Understanding MFCC
- [Mel-Frequency Cepstral Coefficients Explained](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
- [Audio Feature Extraction with Librosa](https://librosa.org/doc/main/feature.html)

### Deep Learning Concepts
- [Understanding Dropout Regularization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout)
- [Batch Normalization in Neural Networks](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)

## 🤝 Contributing

Feel free to contribute to this project by:
- Adding more animal categories
- Improving the model architecture
- Implementing additional audio preprocessing techniques
- Adding more sophisticated evaluation metrics

## 📄 License

This project is open-source and available under the MIT License.

---

**Happy Classifying! 🎉**

*Built with ❤️ using TensorFlow, Librosa, and lots of animal sounds*
