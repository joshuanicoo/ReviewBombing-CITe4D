# analyze_all_movies.py
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
import glob

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
        
        # Load model and tokenizer only if needed for re-analysis
        self.model = None
        self.tokenizer = None
        print("✓ Analyzer initialized (model will be loaded if needed)")
    
    def load_model_if_needed(self):
        """Load model only when needed"""
        if self.model is None:
            print("\nLoading model and tokenizer...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("raphgonda/FilipinoShopping")
                self.model = TFAutoModelForSequenceClassification.from_pretrained("raphgonda/FilipinoShopping")
                print("✓ Model loaded successfully")
            except Exception as e:
                print(f"✗ Error loading model: {e}")
                raise
    
    def analyze_movie_file(self, file_path, reanalyze=False):
        """
        Analyze or summarize a movie sentiment CSV file
        
        Parameters:
        -----------
        file_path : str
            Path to the sentiment CSV file
        reanalyze : bool
            If True, re-analyze the text using the model
            If False, just create summary from existing sentiment data
        """
        movie_name = os.path.basename(file_path).replace('_reviews_sentiment.csv', '').replace('.csv', '')
        
        print(f"\n{'='*60}")
        print(f"PROCESSING: {movie_name.replace('_', ' ').title()}")
        print(f"{'='*60}")
        
        # Load data
        try:
            df = pd.read_csv(file_path)
            print(f"✓ Loaded {len(df):,} rows from {os.path.basename(file_path)}")
            
            # Check if sentiment data already exists
            sentiment_columns = ['sentiment', 'sentiment_label', 'sentiment_score', 
                               'positive', 'negative', 'neutral']
            
            has_sentiment = any(col in df.columns for col in sentiment_columns)
            
            if reanalyze or not has_sentiment:
                # Need to re-analyze or analyze for the first time
                self.load_model_if_needed()
                
                # Find text column
                text_column = None
                possible_text_columns = ['sentence', 'review_text', 'text', 'content']
                for col in possible_text_columns:
                    if col in df.columns:
                        text_column = col
                        break
                
                if text_column is None:
                    print(f"✗ No text column found. Available columns: {df.columns.tolist()}")
                    return None
                
                print(f"Analyzing text from column: '{text_column}'")
                
                # Clean text
                df[text_column] = df[text_column].fillna('').astype(str).str.strip()
                
                # Filter out empty texts
                initial_count = len(df)
                df = df[df[text_column] != '']
                if len(df) < initial_count:
                    print(f"  Removed {initial_count - len(df):,} empty texts")
                
                if len(df) == 0:
                    print("✗ No valid texts to analyze")
                    return None
                
                # Predict sentiment in batches
                print(f"Analyzing sentiment for {len(df):,} texts...")
                texts = df[text_column].tolist()
                results = self.predict_sentiment_batch(texts, batch_size=32)
                
                # Add results to dataframe
                for key in results[0].keys():
                    df[key] = [r[key] for r in results]
                
                print("✓ Sentiment analysis completed")
                
            else:
                # Use existing sentiment data
                print("✓ Using existing sentiment data")
                
                # Find sentiment column
                sentiment_col = None
                for col in ['sentiment', 'sentiment_label']:
                    if col in df.columns:
                        sentiment_col = col
                        break
                
                if sentiment_col:
                    print(f"  Using '{sentiment_col}' column for sentiment")
                    df['sentiment'] = df[sentiment_col]
                
                # Standardize sentiment labels
                df['sentiment'] = df['sentiment'].apply(self.standardize_sentiment_label)
            
            # Calculate statistics
            stats = self._calculate_statistics(df, movie_name)
            
            # Save results
            output_dir = "Sentiment_Results"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save analyzed data
            output_file = f"{output_dir}/{movie_name}_analyzed.csv"
            df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"✓ Results saved to: {output_file}")
            
            # Save summary
            summary_file = f"{output_dir}/{movie_name}_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"✓ Summary saved to: {summary_file}")
            
            return stats
            
        except Exception as e:
            print(f"✗ Error processing file: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def standardize_sentiment_label(self, label):
        """Standardize sentiment labels"""
        if pd.isna(label):
            return 'neutral'
        
        label_str = str(label).lower()
        
        if 'pos' in label_str or 'positive' in label_str or '2' in label_str:
            return 'positive'
        elif 'neg' in label_str or 'negative' in label_str or '0' in label_str:
            return 'negative'
        elif 'neu' in label_str or 'neutral' in label_str or '1' in label_str:
            return 'neutral'
        else:
            # Try to interpret numeric values
            try:
                num = float(label_str)
                if num >= 0.6:
                    return 'positive'
                elif num <= 0.4:
                    return 'negative'
                else:
                    return 'neutral'
            except:
                return label_str
    
    def predict_sentiment_batch(self, texts, batch_size=16):
        """Predict sentiment for a list of texts in batches"""
        if isinstance(texts, str):
            texts = [texts]
        
        all_predictions = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing", unit="batch"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="tf",
                truncation=True,
                padding=True,
                max_length=256
            )
            
            # Predict
            outputs = self.model(inputs)
            predictions = tf.nn.softmax(outputs.logits, axis=-1).numpy()
            
            # Process results
            for j, probs in enumerate(predictions):
                pred_idx = np.argmax(probs)
                confidence = probs[pred_idx]
                
                # Map to sentiment
                if pred_idx == 2:  # Assuming index 2 is positive
                    sentiment = 'positive'
                elif pred_idx == 0:  # Assuming index 0 is negative
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                all_predictions.append({
                    'sentiment': sentiment,
                    'confidence': float(confidence),
                    'prob_negative': float(probs[0]),
                    'prob_neutral': float(probs[1]),
                    'prob_positive': float(probs[2])
                })
        
        return all_predictions
    
    def _calculate_statistics(self, df, movie_name):
        """Calculate comprehensive statistics"""
        stats = {
            'movie': movie_name.replace('_', ' ').title(),
            'total_rows': int(len(df)),
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # Check if sentiment column exists
        if 'sentiment' not in df.columns:
            print("✗ No sentiment column found in data")
            return stats
        
        # Standardize sentiment labels
        df['sentiment_std'] = df['sentiment'].apply(self.standardize_sentiment_label)
        
        # Sentiment distribution
        sentiment_counts = df['sentiment_std'].value_counts()
        for sentiment in ['positive', 'neutral', 'negative']:
            count = sentiment_counts.get(sentiment, 0)
            percentage = (count / len(df)) * 100 if len(df) > 0 else 0
            stats[f'{sentiment}_count'] = int(count)
            stats[f'{sentiment}_percentage'] = round(percentage, 2)
        
        # Confidence statistics (if available)
        if 'confidence' in df.columns:
            stats['avg_confidence'] = round(df['confidence'].mean(), 3)
            stats['min_confidence'] = round(df['confidence'].min(), 3)
            stats['max_confidence'] = round(df['confidence'].max(), 3)
            
            # Confidence by sentiment
            for sentiment in ['positive', 'neutral', 'negative']:
                sentiment_df = df[df['sentiment_std'] == sentiment]
                if len(sentiment_df) > 0:
                    stats[f'avg_confidence_{sentiment}'] = round(sentiment_df['confidence'].mean(), 3)
        
        # Rating statistics (if available)
        if 'rating' in df.columns:
            # Try to convert rating to numeric
            try:
                df['rating_numeric'] = pd.to_numeric(df['rating'], errors='coerce')
                rating_df = df[df['rating_numeric'].notna()]
                
                if len(rating_df) > 0:
                    stats['avg_rating'] = round(rating_df['rating_numeric'].mean(), 2)
                    stats['min_rating'] = float(rating_df['rating_numeric'].min())
                    stats['max_rating'] = float(rating_df['rating_numeric'].max())
                    
                    # Rating by sentiment
                    for sentiment in ['positive', 'neutral', 'negative']:
                        sentiment_df = rating_df[rating_df['sentiment_std'] == sentiment]
                        if len(sentiment_df) > 0:
                            stats[f'avg_rating_{sentiment}'] = round(sentiment_df['rating_numeric'].mean(), 2)
            except:
                pass
        
        # Source distribution (if available)
        if 'source' in df.columns:
            source_counts = df['source'].value_counts()
            stats['sources'] = {str(k): int(v) for k, v in source_counts.items()}
        
        # Print summary
        print(f"\n{'─'*40}")
        print(f"SUMMARY: {stats['movie']}")
        print(f"{'─'*40}")
        print(f"Total Rows: {stats['total_rows']:,}")
        
        if 'positive_count' in stats:
            print(f"Positive: {stats.get('positive_count', 0):,} ({stats.get('positive_percentage', 0):.1f}%)")
            print(f"Neutral: {stats.get('neutral_count', 0):,} ({stats.get('neutral_percentage', 0):.1f}%)")
            print(f"Negative: {stats.get('negative_count', 0):,} ({stats.get('negative_percentage', 0):.1f}%)")
        
        if 'avg_confidence' in stats:
            print(f"Average Confidence: {stats['avg_confidence']:.3f}")
        
        if 'avg_rating' in stats:
            print(f"Average Rating: {stats['avg_rating']:.2f}")
        
        return stats

def find_sentiment_files(data_folder="../ReviewBombing-CITe4D/Sentiment Analysis CSV"):
    """Find all sentiment CSV files in the data folder"""
    print(f"\nSearching for sentiment files in: {data_folder}")
    
    # Look for CSV files with "sentiment" in the name
    sentiment_files = []
    
    # First, check for specific files from your list
    file_list = [
        "all_movies_reviews_sentiment.csv",
        "felix_manalo_reviews_sentiment.csv",
        "hayop_ka_reviews_sentiment.csv", 
        "hello_love_goodbye_reviews_sentiment.csv",
        "hows_of_us_reviews_sentiment.csv",
        "maid_in_malacanang_reviews_sentiment.csv",
        "mallari_reviews_sentiment.csv",
        "praybeyt_benjamin_1_reviews_sentiment.csv",
        "praybeyt_benjamin_2_reviews_sentiment.csv",
        "quezon_reviews_sentiment.csv",
        "sunshine_reviews_sentiment.csv",
        "the_kingmaker_reviews_sentiment.csv"
    ]
    
    for file_name in file_list:
        file_path = os.path.join(data_folder, file_name)
        if os.path.exists(file_path):
            sentiment_files.append(file_path)
            print(f"✓ Found: {file_name}")
        else:
            print(f"✗ Missing: {file_name}")
    
    # Also search for any CSV files with "sentiment" in the name
    all_csv_files = glob.glob(os.path.join(data_folder, "*sentiment*.csv"))
    for file_path in all_csv_files:
        if file_path not in sentiment_files:
            file_name = os.path.basename(file_path)
            sentiment_files.append(file_path)
            print(f"✓ Found additional: {file_name}")
    
    return sentiment_files

def create_master_summary(all_stats):
    """Create a master summary CSV of all movies"""
    if not all_stats:
        return
    
    # Convert to DataFrame
    summary_data = []
    for stats in all_stats:
        row = {
            'movie': stats.get('movie', ''),
            'total_rows': stats.get('total_rows', 0),
            'positive_count': stats.get('positive_count', 0),
            'positive_percentage': stats.get('positive_percentage', 0),
            'neutral_count': stats.get('neutral_count', 0),
            'neutral_percentage': stats.get('neutral_percentage', 0),
            'negative_count': stats.get('negative_count', 0),
            'negative_percentage': stats.get('negative_percentage', 0),
            'avg_confidence': stats.get('avg_confidence', 0),
            'avg_rating': stats.get('avg_rating', 0),
            'analysis_date': stats.get('analysis_date', '')
        }
        summary_data.append(row)
    
    df_summary = pd.DataFrame(summary_data)
    
    # Sort by total rows (descending)
    df_summary = df_summary.sort_values('total_rows', ascending=False)
    
    # Save master summary
    output_dir = "Sentiment_Results"
    os.makedirs(output_dir, exist_ok=True)
    
    master_file = f"{output_dir}/all_movies_summary.csv"
    df_summary.to_csv(master_file, index=False, encoding='utf-8')
    
    print(f"\n{'='*60}")
    print(f"MASTER SUMMARY CREATED: {master_file}")
    print(f"{'='*60}")
    print(f"Total movies analyzed: {len(all_stats)}")
    print(f"Total rows analyzed: {df_summary['total_rows'].sum():,}")
    
    # Print summary table
    print("\nSummary Table:")
    display_cols = ['movie', 'total_rows', 'positive_percentage', 'neutral_percentage', 'negative_percentage']
    if 'avg_rating' in df_summary.columns:
        display_cols.append('avg_rating')
    
    print(df_summary[display_cols].to_string(index=False))
    
    # Calculate overall averages
    if len(df_summary) > 0:
        print(f"\nOverall Averages:")
        print(f"Average Positive: {df_summary['positive_percentage'].mean():.1f}%")
        print(f"Average Neutral: {df_summary['neutral_percentage'].mean():.1f}%")
        print(f"Average Negative: {df_summary['negative_percentage'].mean():.1f}%")
        if 'avg_rating' in df_summary.columns and df_summary['avg_rating'].notna().any():
            print(f"Average Rating: {df_summary['avg_rating'].mean():.2f}")

def main():
    """Main analysis function"""
    print("=" * 70)
    print("MOVIE SENTIMENT ANALYSIS - PROCESS EXISTING DATA")
    print("=" * 70)
    
    # Ask user what to do
    print("\nWhat would you like to do?")
    print("1. Create summaries from existing sentiment data (no re-analysis)")
    print("2. Re-analyze all files with the model")
    print("3. Re-analyze only files without sentiment data")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    if choice == '1':
        reanalyze_all = False
        reanalyze_missing = False
        print("Creating summaries from existing sentiment data...")
    elif choice == '2':
        reanalyze_all = True
        reanalyze_missing = False
        print("Re-analyzing all files with the model...")
    elif choice == '3':
        reanalyze_all = False
        reanalyze_missing = True
        print("Re-analyzing only files without sentiment data...")
    else:
        print("Invalid choice. Defaulting to option 1.")
        reanalyze_all = False
        reanalyze_missing = False
    
    # Initialize analyzer
    analyzer = MovieSentimentAnalyzer()
    
    # Find all sentiment files
    sentiment_files = find_sentiment_files()
    
    if not sentiment_files:
        print("\n✗ No sentiment files found!")
        print("Please make sure your CSV files are in the current directory or specify the correct path.")
        return
    
    print(f"\nFound {len(sentiment_files)} sentiment files to process")
    print("Starting processing...\n")
    
    # Process each file
    all_stats = []
    for i, file_path in enumerate(sentiment_files, 1):
        print(f"\n[{i}/{len(sentiment_files)}] ", end="")
        
        # Determine if we need to reanalyze this file
        if reanalyze_all:
            reanalyze = True
        elif reanalyze_missing:
            # Check if file has sentiment data
            try:
                df = pd.read_csv(file_path, nrows=1)
                has_sentiment = any(col in df.columns for col in ['sentiment', 'sentiment_label'])
                reanalyze = not has_sentiment
            except:
                reanalyze = True
        else:
            reanalyze = False
        
        stats = analyzer.analyze_movie_file(file_path, reanalyze=reanalyze)
        if stats:
            all_stats.append(stats)
    
    # Create master summary
    if all_stats:
        create_master_summary(all_stats)
        
        # Also save as JSON for visualization
        master_json = "Sentiment_Results/all_movies_summary.json"
        with open(master_json, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)
        print(f"✓ Detailed JSON summary: {master_json}")
    
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE!")
    print("=" * 70)
    print("\nResults saved in the 'Sentiment_Results' folder:")
    print("- Individual analyzed CSV files")
    print("- Individual summary JSON files")
    print("- Master summary CSV file")
    print("- Master summary JSON file")

if __name__ == "__main__":
    main()