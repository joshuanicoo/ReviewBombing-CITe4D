# analyze_movie_sentiment.py
import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import json

class MovieSentimentAnalyzer:
    def __init__(self, model_path="raphgonda/FilipinoShopping"):
        """Initialize the analyzer with the FilipinoShopping model"""
        print("=" * 70)
        print("MOVIE REVIEW SENTIMENT ANALYZER")
        print("=" * 70)
        print(f"Model: {model_path}")
        
        # Check device (for TensorFlow)
        self.device = "/GPU:0" if len(tf.config.list_physical_devices('GPU')) > 0 else "/CPU:0"
        print(f"Device: {self.device}")
        
        # Load model and tokenizer
        print("\nLoading model and tokenizer...")
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load TensorFlow model
            self.model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
            
            print("✓ Model loaded successfully")
            
            # Get label mapping
            self.label_mapping = self._get_label_mapping()
            print(f"✓ Label mapping: {self.label_mapping}")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
        
        print("✓ Analyzer initialized successfully")
    
    def _get_label_mapping(self):
        """Get label mapping from model config"""
        try:
            config = self.model.config
            if hasattr(config, 'id2label') and config.id2label:
                # Convert to simpler mapping
                mapping = {}
                for key, value in config.id2label.items():
                    # Standardize labels
                    label_lower = value.lower()
                    if 'neg' in label_lower:
                        mapping[f"LABEL_{key}"] = 'negative'
                    elif 'pos' in label_lower:
                        mapping[f"LABEL_{key}"] = 'positive'
                    elif 'neu' in label_lower:
                        mapping[f"LABEL_{key}"] = 'neutral'
                    else:
                        mapping[f"LABEL_{key}"] = label_lower
                return mapping
        except:
            pass
        
        # Default mapping based on your config
        return {
            'LABEL_0': 'negative',
            'LABEL_1': 'neutral', 
            'LABEL_2': 'positive'
        }
    
    def predict_sentiment(self, texts, batch_size=8):
        """Predict sentiment for a list of texts"""
        if isinstance(texts, str):
            texts = [texts]
        
        all_predictions = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="tf",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Predict
            outputs = self.model(inputs)
            predictions = tf.nn.softmax(outputs.logits, axis=-1).numpy()
            
            # Process results
            for j, probs in enumerate(predictions):
                pred_idx = np.argmax(probs)
                confidence = probs[pred_idx]
                label_id = f"LABEL_{pred_idx}"
                label = self.label_mapping.get(label_id, label_id)
                
                # Standardize labels
                if 'neg' in label.lower():
                    label = 'negative'
                elif 'pos' in label.lower():
                    label = 'positive'
                elif 'neu' in label.lower():
                    label = 'neutral'
                
                all_predictions.append({
                    'label': label,
                    'score': float(confidence),
                    'probabilities': probs.tolist()
                })
        
        return all_predictions
    
    def analyze_movie_file(self, file_path):
        """Analyze a single movie CSV file"""
        movie_name = os.path.basename(file_path).replace('_reviews.csv', '').replace('.csv', '')
        
        print(f"\n{'='*60}")
        print(f"ANALYZING: {movie_name.replace('_', ' ').title()}")
        print(f"{'='*60}")
        
        # Load data
        try:
            df = pd.read_csv(file_path)
            print(f"✓ Loaded {len(df)} reviews")
            
            # Check required columns
            required_cols = ['review_text']
            for col in required_cols:
                if col not in df.columns:
                    print(f"✗ Required column '{col}' not found!")
                    print(f"  Available columns: {df.columns.tolist()}")
                    return None
            
            # Clean text
            df['review_text'] = df['review_text'].fillna('').astype(str)
            
            # Filter out empty reviews
            initial_count = len(df)
            df = df[df['review_text'].str.strip() != '']
            if len(df) < initial_count:
                print(f"  Removed {initial_count - len(df)} empty reviews")
            
            if len(df) == 0:
                print("✗ No valid reviews to analyze")
                return None
            
            # Predict sentiment
            print(f"Analyzing sentiment for {len(df)} reviews...")
            texts = df['review_text'].tolist()
            results = self.predict_sentiment(texts)
            
            # Add results to dataframe
            df['sentiment_label'] = [r['label'] for r in results]
            df['sentiment_score'] = [r['score'] for r in results]
            
            # Map to standardized labels
            def standardize_label(label):
                label_lower = str(label).lower()
                if 'pos' in label_lower or '2' in label_lower:
                    return 'positive'
                elif 'neg' in label_lower or '0' in label_lower:
                    return 'negative'
                elif 'neu' in label_lower or '1' in label_lower:
                    return 'neutral'
                else:
                    return label_lower
            
            df['sentiment'] = df['sentiment_label'].apply(standardize_label)
            
            # Calculate statistics
            stats = self._calculate_statistics(df, movie_name)
            
            # Save results
            output_file = f"{movie_name}_analyzed.csv"
            df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"✓ Results saved to: {output_file}")
            
            # Save summary
            summary_file = f"{movie_name}_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
            print(f"✓ Summary saved to: {summary_file}")
            
            return stats
            
        except Exception as e:
            print(f"✗ Error analyzing file: {e}")
            return None
    
    def _calculate_statistics(self, df, movie_name):
        """Calculate comprehensive statistics"""
        stats = {
            'movie': movie_name,
            'total_reviews': len(df),
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        for sentiment in ['positive', 'neutral', 'negative']:
            count = sentiment_counts.get(sentiment, 0)
            percentage = (count / len(df)) * 100 if len(df) > 0 else 0
            stats[f'{sentiment}_count'] = int(count)
            stats[f'{sentiment}_percentage'] = round(percentage, 2)
        
        # Confidence statistics
        if 'sentiment_score' in df.columns:
            stats['avg_confidence'] = round(df['sentiment_score'].mean(), 3)
            stats['min_confidence'] = round(df['sentiment_score'].min(), 3)
            stats['max_confidence'] = round(df['sentiment_score'].max(), 3)
        
        # Rating statistics (if available)
        if 'rating' in df.columns:
            stats['avg_rating'] = round(df['rating'].mean(), 1)
            stats['min_rating'] = int(df['rating'].min())
            stats['max_rating'] = int(df['rating'].max())
        
        # Source distribution (if available)
        if 'source' in df.columns:
            source_counts = df['source'].value_counts()
            stats['sources'] = {str(k): int(v) for k, v in source_counts.items()}
        
        # Print summary
        print(f"\n{'─'*40}")
        print(f"SUMMARY: {movie_name.replace('_', ' ').title()}")
        print(f"{'─'*40}")
        print(f"Total Reviews: {stats['total_reviews']}")
        print(f"Positive: {stats.get('positive_count', 0)} ({stats.get('positive_percentage', 0):.1f}%)")
        print(f"Neutral: {stats.get('neutral_count', 0)} ({stats.get('neutral_percentage', 0):.1f}%)")
        print(f"Negative: {stats.get('negative_count', 0)} ({stats.get('negative_percentage', 0):.1f}%)")
        
        if 'avg_confidence' in stats:
            print(f"Average Confidence: {stats['avg_confidence']:.3f}")
        
        if 'avg_rating' in stats:
            print(f"Average Rating: {stats['avg_rating']:.1f}/100")
        
        return stats

# Test function
def test_analyzer():
    """Test the analyzer"""
    print("Testing MovieSentimentAnalyzer...")
    
    # Initialize analyzer
    analyzer = MovieSentimentAnalyzer()
    
    # Test with sample texts
    test_texts = [
        "Ang ganda ng pelikula! Sobrang galing ng mga artista.",
        "Hindi ko nagustuhan, boring ang kwento.",
        "Sakto lang, okay naman pero walang wow factor.",
        "Napakaganda ng cinematography, world-class!"
    ]
    
    print("\nTesting predictions:")
    results = analyzer.predict_sentiment(test_texts)
    
    for text, result in zip(test_texts, results):
        print(f"  '{text[:40]}...' -> {result['label']} ({result['score']:.3f})")

if __name__ == "__main__":
    test_analyzer()