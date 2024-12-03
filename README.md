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

## Note
Make sure to have all required dependencies installed and the correct file paths set up before running the preprocessing pipeline.
