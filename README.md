# Image Caption Generator - Data Preprocessing

This repository contains the preprocessing steps for an Image Caption Generator project. The preprocessing pipeline includes image processing, feature extraction, caption cleaning, and vocabulary building.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Image Preprocessing](#image-preprocessing)
- [Feature Extraction](#feature-extraction)
- [Caption Processing](#caption-processing)
- [Vocabulary Building](#vocabulary-building)

## Prerequisites

```python
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
import string
from collections import Counter
import os
```

## Image Preprocessing

The image preprocessing function handles the following steps:
- Loads and resizes images to 299x299 pixels (InceptionV3 requirement)
- Converts images to arrays
- Expands dimensions for batch processing
- Applies InceptionV3 preprocessing

```python
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array
```

## Feature Extraction

Features are extracted using the InceptionV3 model:
- Processes each image through the preprocessing pipeline
- Extracts 2048-dimensional feature vectors
- Stores features with corresponding image IDs

```python
def extract_features(image_paths):
    features = {}
    for image_path in image_paths:
        image_id = os.path.split(image_path)[1]
        img = preprocess_image(image_path)
        feature = model.predict(img, verbose=0)
        feature = feature.reshape((2048,))
        features[image_id] = feature
    return features
```

## Caption Processing

Caption processing includes:
1. Loading captions from file:
```python
def load_captions(captions_file):
    # Loads captions from CSV file
    # Returns dictionary with image IDs as keys and captions as values
```

2. Cleaning captions:
```python
def clean_captions(captions):
    # Converts to lowercase
    # Removes special characters and numbers
    # Adds start and end tokens
    # Returns cleaned captions dictionary
```

## Vocabulary Building

The vocabulary building process:
- Counts word frequencies
- Filters words based on minimum threshold
- Adds special tokens (<start>, <end>, <unk>, <pad>)
- Creates word-to-index and index-to-word mappings

```python
def build_vocabulary(cleaned_captions, threshold=5):
    # Creates vocabulary from cleaned captions
    # Returns vocabulary list and mapping dictionaries
```

## Caption Tokenization

Tokenizes the processed captions:
```python
def tokenize_captions(captions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    return tokenizer
```

## Usage

1. Preprocess images:
```python
image_features = extract_features(image_paths)
```

2. Process captions:
```python
captions = load_captions('captions.csv')
cleaned_captions = clean_captions(captions)
```

3. Build vocabulary:
```python
vocabulary, word_to_idx, idx_to_word = build_vocabulary(cleaned_captions)
```

4. Tokenize captions:
```python
tokenizer = tokenize_captions(all_captions)
```

# Image Caption Generator - Model Training

This section describes the model architecture, training process, and caption generation for the Image Caption Generator project.

## Table of Contents
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Data Generation](#data-generation)
- [Caption Generation](#caption-generation)
- [Usage Example](#usage-example)

## Model Architecture

The model uses a combination of CNN features and LSTM for caption generation:

```python
def build_caption_generator(vocab_size, max_length, embedding_dim=256, rnn_units=512):
```

Key components:
- Image feature input (2048-dimensional vector)
- Caption input sequence
- Word embedding layer
- LSTM layer with dropout
- Dense layer for word prediction

Architecture details:
- Embedding dimension: 256
- LSTM units: 512
- Dropout rate: 0.5
- Activation: Softmax
- Loss: Sparse categorical crossentropy
- Optimizer: Adam (learning rate: 0.01)

## Training Process

The training process is managed by the `train_model` function:

```python
def train_model(train_captions, train_features, word_to_idx,
                max_length=40, batch_size=32, epochs=5):
```

Training parameters:
- Maximum sequence length: 40
- Batch size: 32
- Number of epochs: 5
- Steps per epoch: Calculated based on total captions

## Data Generation

The data generator creates training batches on-the-fly:

```python
def data_generator(captions_dict, features_dict, word_to_idx, max_length, batch_size):
```

Features:
- Randomly shuffles image IDs
- Generates batches of image features and caption sequences
- Handles padding of sequences
- Creates input-output pairs for training

## Caption Generation

Caption generation process:

```python
def generate_caption(model, image_features, word_to_idx, idx_to_word, max_length):
```

Generation process:
1. Starts with '<start>' token
2. Iteratively predicts next word
3. Stops when '<end>' token is predicted
4. Handles unknown words with '<unk>' token

## Usage Example

1. Train the model:
```python
max_length = 40
batch_size = 32
epochs = 5

model, history = train_model(
    train_captions,
    train_features,
    word_to_idx,
    max_length,
    batch_size,
    epochs
)
```

2. Save the trained model:
```python
model.save('caption_generator_model.keras')
```

3. Generate captions:
```python
caption = generate_caption(
    model,
    image_features,
    word_to_idx,
    idx_to_word,
    max_length
)
```

## Model Parameters

Default hyperparameters:
- Embedding dimension: 256
- RNN units: 512
- Maximum sequence length: 40
- Batch size: 32
- Training epochs: 5
- Learning rate: 0.01

## Notes

- The model uses teacher forcing during training
- Dropout (0.5) is applied to prevent overfitting
- The generator ensures efficient memory usage during training
- The model can be fine-tuned by adjusting hyperparameters

Remember to monitor training progress and adjust hyperparameters as needed for optimal performance.

# Future Improvements and Enhancements

## Real-Time Captioning Implementation

### Planned Features
- Integration with webcam feed for live video captioning
- Real-time image processing pipeline
- Low-latency caption generation
- Streaming capability for continuous captioning

### Technical Requirements
```python
# Future implementation example
def real_time_captioning():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        
        # Process frame
        processed_frame = preprocess_image(frame)
        
        # Generate caption
        caption = generate_caption(model, processed_frame, word_to_idx, idx_to_word, max_length)
        
        # Display result
        # Release resources when done
```

## Testing Improvements

### Planned Enhancements

1. **Evaluation Metrics**
   - BLEU score implementation
   - METEOR score
   - CIDEr score
   - ROUGE-L score

```python
# Example of future evaluation implementation
def evaluate_model(model, test_data):
    """
    Comprehensive model evaluation
    """
    bleu_scores = calculate_bleu(predictions, references)
    meteor_score = calculate_meteor(predictions, references)
    cider_score = calculate_cider(predictions, references)
    rouge_score = calculate_rouge(predictions, references)
    
    return {
        'bleu': bleu_scores,
        'meteor': meteor_score,
        'cider': cider_score,
        'rouge': rouge_score
    }
```

