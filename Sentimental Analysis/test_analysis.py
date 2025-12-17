# test_analysis.py - Quick test before full analysis
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import numpy as np

print("=== QUICK TEST OF MODEL AND DATA ===")

# 1. Test model loading
print("\n1. Testing model loading...")
try:
    model_path = "."  # Current directory
    
    # Check what files are available
    print("Checking available model files...")
    if os.path.exists("tf_model.h5"):
        print("✓ Found tf_model.h5 (TensorFlow model)")
    if os.path.exists("config.json"):
        print("✓ Found config.json")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("✓ Tokenizer loaded")
    
    # Load TensorFlow model - NO from_tf=True for TFAutoModel
    print("Loading TensorFlow model...")
    model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
    print("✓ TensorFlow model loaded successfully")
    
    # Check label mapping from config
    label_mapping = model.config.id2label
    print(f"✓ Label mapping: {label_mapping}")
    
    # Test predictions
    print("\n2. Testing predictions:")
    test_texts = [
        "Ang ganda ng pelikula! Sobrang galing ng mga artista.",
        "Hindi ko nagustuhan, boring ang kwento.",
        "Sakto lang, okay naman pero walang wow factor.",
        "Napakaganda ng cinematography, world-class!"
    ]
    
    for text in test_texts:
        inputs = tokenizer(
            text,
            return_tensors="tf",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        outputs = model(inputs)
        predictions = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
        pred_idx = np.argmax(predictions)
        confidence = predictions[pred_idx]
        
        # Get label from mapping
        label = label_mapping.get(pred_idx, f"LABEL_{pred_idx}")
        
        print(f"\n  Text: '{text[:50]}...'")
        print(f"  → Sentiment: {label}")
        print(f"  → Confidence: {confidence:.3f}")
        print(f"  → Probabilities: [Neg: {predictions[0]:.3f}, Neu: {predictions[1]:.3f}, Pos: {predictions[2]:.3f}]")
    
except Exception as e:
    print(f"✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()

# 2. Test data loading (same as before)
print("\n3. Testing data loading...")
test_files = ['../ReviewBombing-CITe4D/Cleaned Data/reviews.csv', 
              '../ReviewBombing-CITe4D/Cleaned Data/hello_love_goodbye_reviews.csv']

for file in test_files:
    try:
        if os.path.exists(file):
            df = pd.read_csv(file)
            print(f"\n✓ {file}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {df.columns.tolist()}")
            
            # Show sample
            if 'review_text' in df.columns:
                sample = df['review_text'].iloc[0] if len(df) > 0 else "No data"
                print(f"  Sample review: {str(sample)[:100]}...")
            
            if 'rating' in df.columns and len(df) > 0:
                print(f"  Rating range: {df['rating'].min():.0f}-{df['rating'].max():.0f}")
            
            if 'source' in df.columns and len(df) > 0:
                print(f"  Sources: {df['source'].unique().tolist()[:5]}")  # Show first 5
        else:
            print(f"\n✗ File not found: {file}")
            
    except Exception as e:
        print(f"✗ Could not load {file}: {e}")

# 4. Test batch prediction
print("\n4. Testing batch prediction...")
try:
    # Create a simple test with movie reviews
    test_reviews = [
        "This movie is fantastic! I loved every minute of it.",
        "Terrible film, wasted my time and money.",
        "It was okay, nothing special but not bad either.",
        "One of the best movies I've seen this year!",
        "Boring and predictable, wouldn't recommend."
    ]
    
    print(f"Testing {len(test_reviews)} sample reviews...")
    
    # Tokenize batch
    inputs = tokenizer(
        test_reviews,
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    # Predict batch
    outputs = model(inputs)
    predictions = tf.nn.softmax(outputs.logits, axis=-1).numpy()
    
    for i, (review, probs) in enumerate(zip(test_reviews, predictions)):
        pred_idx = np.argmax(probs)
        label = label_mapping.get(pred_idx, f"LABEL_{pred_idx}")
        confidence = probs[pred_idx]
        
        print(f"\n  Review {i+1}: '{review[:40]}...'")
        print(f"    → {label} ({confidence:.3f})")
    
except Exception as e:
    print(f"✗ Batch test error: {e}")