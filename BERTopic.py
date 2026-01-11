from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import numpy as np
import os
import json
import re
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict


# ============================
# 1. COMBINED CLASS (Sentiment + Topic)
# ============================
class MovieTopicSentimentAnalyzer:
    def __init__(self, sentiment_model_path="raphgonda/FilipinoShopping"):
        """Initialize both sentiment and topic analysis"""
        print("=" * 70)
        print("MOVIE TOPIC & SENTIMENT ANALYZER")
        print("=" * 70)
        print("Mode: PER-SENTENCE SENTIMENT ANALYSIS")
        print("=" * 70)

        # Initialize sentiment analyzer
        self.sentiment_analyzer = self._init_sentiment_analyzer(sentiment_model_path)

        # Initialize embedding model for topics
        print("\nLoading embedding model for topic modeling...")
        self.embedding_model = SentenceTransformer(
            "paraphrase-multilingual-mpnet-base-v2",
            device='cuda' if tf.config.list_physical_devices('GPU') else 'cpu'
        )
        print("✓ Embedding model loaded")

        # Stopwords
        self.custom_stopwords = self._get_stopwords()

        print("✓ Analyzer initialized successfully")

    def _split_into_sentences(self, text):
        """Split text into sentences using regex."""
        if not isinstance(text, str) or not text.strip():
            return []

        # Clean the text
        text = text.strip()

        # Better sentence splitting regex that handles abbreviations
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

        # If the above didn't work well, use a simpler approach
        if len(sentences) <= 1:
            sentences = re.split(r'[.!?]+', text)

        # Filter out very short sentences and whitespace
        filtered = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) >= 10:  # At least 10 characters
                # Add period if missing
                if not sentence.endswith(('.', '!', '?')):
                    sentence = sentence + '.'
                filtered.append(sentence)

        return filtered

    def predict_sentiment_batch(self, texts, batch_size=8):
        """Predict sentiment for a batch of texts (sentences)"""
        if isinstance(texts, str):
            texts = [texts]

        all_predictions = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing sentiment"):
            batch_texts = texts[i:i + batch_size]

            # Skip empty texts
            batch_texts = [text for text in batch_texts if text and str(text).strip()]
            if not batch_texts:
                continue

            # Tokenize
            inputs = self.sentiment_analyzer['tokenizer'](
                batch_texts,
                return_tensors="tf",
                truncation=True,
                padding=True,
                max_length=512
            )

            # Predict
            outputs = self.sentiment_analyzer['model'](inputs)
            predictions = tf.nn.softmax(outputs.logits, axis=-1).numpy()

            # Process results
            for j, probs in enumerate(predictions):
                pred_idx = np.argmax(probs)
                confidence = probs[pred_idx]
                label_id = f"LABEL_{pred_idx}"
                label = self.sentiment_analyzer['label_mapping'].get(label_id, label_id)

                # Standardize labels
                label_lower = label.lower()
                if 'neg' in label_lower:
                    label = 'negative'
                elif 'pos' in label_lower:
                    label = 'positive'
                elif 'neu' in label_lower:
                    label = 'neutral'

                all_predictions.append({
                    'text': batch_texts[j],
                    'sentiment_label': label,
                    'sentiment_score': float(confidence),
                    'sentiment_confidence': float(confidence)
                })

        return all_predictions

    def analyze_sentences_per_review(self, reviews):
        """Analyze sentiment for each sentence in each review"""
        print("\nPerforming sentence-level sentiment analysis...")

        all_sentence_results = []
        review_sentence_mapping = []

        for review_idx, review_text in enumerate(tqdm(reviews, desc="Processing reviews")):
            # Split review into sentences
            sentences = self._split_into_sentences(review_text)

            if not sentences:
                # If no valid sentences, add placeholder
                review_sentence_mapping.append({
                    'review_index': review_idx,
                    'sentence_count': 0,
                    'sentence_indices': [],
                    'aggregated_sentiment': {
                        'sentiment_label': 'neutral',
                        'sentiment_score': 0.0
                    }
                })
                continue

            # Get predictions for all sentences in this review
            sentence_predictions = self.predict_sentiment_batch(sentences)

            # Store sentence-level results
            sentence_start_idx = len(all_sentence_results)
            for sent_idx, (sentence, pred) in enumerate(zip(sentences, sentence_predictions)):
                sentence_result = {
                    'review_index': review_idx,
                    'sentence_index': sent_idx,
                    'sentence_text': sentence,
                    'sentiment_label': pred['sentiment_label'],
                    'sentiment_score': pred['sentiment_score'],
                    'sentiment_confidence': pred['sentiment_confidence']
                }
                all_sentence_results.append(sentence_result)

            # Calculate aggregated review sentiment
            sentence_labels = [pred['sentiment_label'] for pred in sentence_predictions]
            sentence_scores = [pred['sentiment_score'] for pred in sentence_predictions]

            # Count sentiment distribution
            from collections import Counter
            label_counts = Counter(sentence_labels)

            # Determine dominant sentiment (majority vote)
            if sentence_labels:
                dominant_label = max(label_counts.items(), key=lambda x: x[1])[0]
                avg_score = np.mean(sentence_scores)
            else:
                dominant_label = 'neutral'
                avg_score = 0.0

            # Store review-level aggregation
            review_sentence_mapping.append({
                'review_index': review_idx,
                'sentence_count': len(sentences),
                'sentence_indices': list(range(sentence_start_idx, sentence_start_idx + len(sentences))),
                'aggregated_sentiment': {
                    'sentiment_label': dominant_label,
                    'sentiment_score': float(avg_score)
                },
                'sentiment_distribution': dict(label_counts)
            })

        return all_sentence_results, review_sentence_mapping

    def _init_sentiment_analyzer(self, model_path):
        """Initialize sentiment analyzer"""
        print(f"Sentiment Model: {model_path}")
        device = "/GPU:0" if len(tf.config.list_physical_devices('GPU')) > 0 else "/CPU:0"
        print(f"Device: {device}")

        print("Loading sentiment model and tokenizer...")
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Load TensorFlow model
            model = TFAutoModelForSequenceClassification.from_pretrained(model_path)

            # Get label mapping
            config = model.config
            if hasattr(config, 'id2label') and config.id2label:
                label_mapping = {}
                for key, value in config.id2label.items():
                    label_lower = value.lower()
                    if 'neg' in label_lower:
                        label_mapping[f"LABEL_{key}"] = 'negative'
                    elif 'pos' in label_lower:
                        label_mapping[f"LABEL_{key}"] = 'positive'
                    elif 'neu' in label_lower:
                        label_mapping[f"LABEL_{key}"] = 'neutral'
                    else:
                        label_mapping[f"LABEL_{key}"] = label_lower
            else:
                label_mapping = {
                    'LABEL_0': 'negative',
                    'LABEL_1': 'neutral',
                    'LABEL_2': 'positive'
                }

            print("✓ Sentiment model loaded successfully")

            return {
                'tokenizer': tokenizer,
                'model': model,
                'label_mapping': label_mapping
            }

        except Exception as e:
            print(f"✗ Error loading sentiment model: {e}")
            raise

    def _get_stopwords(self):
        """Get custom stopwords"""
        filipino_stopwords = [
            'ang', 'ng', 'mga', 'sa', 'yung', 'yang', 'yon', 'yong', 'ito', 'iyan', 'iyon',
            'si', 'sina', 'ni', 'nina', 'kay', 'kina', 'ako', 'ko', 'ikaw', 'ka', 'mo',
            'kami', 'kita', 'tayo', 'sila', 'siya', 'namin', 'natin', 'ninyo', 'nila',
            'akin', 'amin', 'atin', 'inyo', 'kanila', 'kanya', 'kanyang', 'at', 'o', 'ni',
            'para', 'kung', 'pero', 'dahil', 'kasi', 'basta', 'lang', 'din', 'rin', 'daw',
            'raw', 'man', 'pala', 'yata', 'nga', 'naman', 'talaga', 'parang', 'siguro',
            'marahil', 'ay', 'na', 'pa', 'ba', 'ho', 'po', 'ano', 'sino', 'saan', 'kailan',
            'bakit', 'paano', 'alin', 'magkano', 'kumusta', 'dito', 'diyan', 'doon',
            'ngayon', 'kahapon', 'bukas', 'mamaya', 'kanina', 'noon', 'may', 'mayroon',
            'wala', 'walang', 'meron', 'lahat', 'ilan', 'marami', 'kaunti', 'konti', 'isa',
            'dalawa', 'tatlo', 'apat', 'lima', 'pareho', 'iba', 'eh', 'ayun', 'ayos',
            'sige', 'oke', 'okay', 'ah', 'oh', 'ha', 'noh', 'di', 'mag', 'nag', 'pag',
            'um', 'in', 'hin', 'an', 'han', 'ma', 'pa', 'ka', 'maka', 'naka', 'paka',
            'ganun', 'hindi', 'dapat', 'niya', 'sobrang', 'nung', 'jo', 'pagod', 'jonathan', 'johnrey', 'john',
            'rey', 'john rey', 'babae', 'isang', 'movie', 'filipino', 'philippine',
            'philippine cinema', 'palibhasa', 'ngiti', 'ulit', 'gets',
            'sunshine', 'pelikula', 'sana', 'yan', 'yap', 'richard',
            'praybeyt', 'nimfa', 'nang', 'mas', 'kahit', 'alam', 'nay',
            'tay', 'la', 'la la', 'la land', 'baka', 'baka maakay', 'hanggang', 'hanggang leeg', 'leeg'
        ]

        return filipino_stopwords + list(ENGLISH_STOP_WORDS)

    def analyze_movie(self, movie_name, df):
        """Analyze a movie with both topic modeling and sentiment analysis"""
        print(f"\n{'=' * 60}")
        print(f"ANALYZING: {movie_name}")
        print(f"{'=' * 60}")

        # Prepare data
        docs = df["review_text"].astype(str).tolist()
        dates = df["date"].tolist() if "date" in df.columns else None
        doc_count = len(docs)

        if doc_count < 10:
            print(f"Skipping {movie_name} — NOT ENOUGH REVIEWS ({doc_count})")
            return None

        # Create model directory
        model_dir = f"./Models/{movie_name.replace(' ', '_')}"
        os.makedirs(model_dir, exist_ok=True)

        # ========================
        # 2. SENTENCE-LEVEL SENTIMENT ANALYSIS
        # ========================
        print("\n1. Performing SENTENCE-LEVEL sentiment analysis...")
        sentence_results, review_mapping = self.analyze_sentences_per_review(docs)

        # Convert to DataFrames for easier manipulation
        sentence_df = pd.DataFrame(sentence_results)
        review_mapping_df = pd.DataFrame(review_mapping)

        # Save sentence-level results
        sentence_df.to_csv(f"{model_dir}/sentence_sentiment_results.csv",
                           index=False, encoding="utf-8-sig")
        review_mapping_df.to_csv(f"{model_dir}/review_sentence_mapping.csv",
                                 index=False, encoding="utf-8-sig")

        print(f"✓ Analyzed {len(sentence_df)} sentences from {len(review_mapping_df)} reviews")

        # Add aggregated sentiment to main dataframe
        df = df.copy()
        aggregated_labels = []
        aggregated_scores = []

        for mapping in review_mapping:
            aggregated_labels.append(mapping['aggregated_sentiment']['sentiment_label'])
            aggregated_scores.append(mapping['aggregated_sentiment']['sentiment_score'])

        df["sentiment_label"] = aggregated_labels
        df["sentiment_score"] = aggregated_scores

        # Calculate overall sentiment statistics
        sentiment_counts = df["sentiment_label"].value_counts()
        total_reviews = len(df)

        print(f"\n✓ Review-level Sentiment Distribution (aggregated from sentences):")
        for sentiment in ['positive', 'neutral', 'negative']:
            count = sentiment_counts.get(sentiment, 0)
            percentage = (count / total_reviews) * 100
            print(f"  {sentiment.title()}: {count} ({percentage:.1f}%)")

        # Also show sentence-level statistics
        print(f"\n✓ Sentence-level Sentiment Distribution:")
        sentence_counts = sentence_df["sentiment_label"].value_counts()
        total_sentences = len(sentence_df)
        for sentiment in ['positive', 'neutral', 'negative']:
            count = sentence_counts.get(sentiment, 0)
            percentage = (count / total_sentences) * 100
            print(f"  {sentiment.title()}: {count} ({percentage:.1f}%)")

        # ========================
        # 3. TOPIC MODELING
        # ========================
        print("\n2. Performing topic modeling...")

        # Determine topic parameters
        if doc_count < 200:
            min_topic_size = 3
        elif doc_count < 1000:
            min_topic_size = 10
        else:
            min_topic_size = 20

        min_df_value = max(1, int(doc_count * 0.001))
        safe_min_df = min(min_df_value, max(1, int(min_topic_size * 0.5)))
        max_df_value = max(safe_min_df + 1, int(doc_count * 0.8))

        print(f"  Parameters: min_topic_size={min_topic_size}, min_df={safe_min_df}")

        # Vectorizer
        vectorizer_model = CountVectorizer(
            stop_words=self.custom_stopwords,
            ngram_range=(1, 2),
            min_df=safe_min_df,
            max_df=max_df_value
        )

        # Create BERTopic model
        topic_model = BERTopic(
            embedding_model=self.embedding_model,
            vectorizer_model=vectorizer_model,
            language="multilingual",
            min_topic_size=min_topic_size,
            verbose=False
        )

        # Fit the model
        topics, probs = topic_model.fit_transform(docs)

        # Add topics to dataframe
        df["topic"] = topics

        # ========================
        # 4. COMBINE TOPIC & SENTIMENT
        # ========================
        print("\n3. Analyzing sentiment per topic...")

        # Group by topic and analyze sentiment
        topic_sentiment_stats = {}

        for topic in set(topics):
            if topic == -1:  # Skip outliers
                continue

            topic_docs = df[df["topic"] == topic]
            topic_size = len(topic_docs)

            if topic_size < 2:  # Skip very small topics
                continue

            # Get sentiment distribution for this topic
            sentiment_dist = topic_docs["sentiment_label"].value_counts()

            # Calculate percentages
            total_topic_reviews = topic_size
            positive_count = sentiment_dist.get('positive', 0)
            negative_count = sentiment_dist.get('negative', 0)
            neutral_count = sentiment_dist.get('neutral', 0)

            positive_pct = (positive_count / total_topic_reviews) * 100
            negative_pct = (negative_count / total_topic_reviews) * 100
            neutral_pct = (neutral_count / total_topic_reviews) * 100

            # Determine dominant sentiment
            if positive_pct > 50:
                dominant_sentiment = "POSITIVE"
            elif negative_pct > 50:
                dominant_sentiment = "NEGATIVE"
            elif neutral_pct > 50:
                dominant_sentiment = "NEUTRAL"
            else:
                # No clear majority
                if positive_pct >= negative_pct and positive_pct >= neutral_pct:
                    dominant_sentiment = "MOSTLY_POSITIVE"
                elif negative_pct >= positive_pct and negative_pct >= neutral_pct:
                    dominant_sentiment = "MOSTLY_NEGATIVE"
                else:
                    dominant_sentiment = "MIXED"

            # Get topic keywords
            topic_keywords = topic_model.get_topic(topic)
            top_keywords = [word for word, score in topic_keywords[:5]]

            # Calculate average sentiment score
            avg_sentiment_score = topic_docs["sentiment_score"].mean()

            topic_sentiment_stats[topic] = {
                'topic_size': topic_size,
                'dominant_sentiment': dominant_sentiment,
                'positive_count': int(positive_count),
                'positive_percentage': round(positive_pct, 2),
                'negative_count': int(negative_count),
                'negative_percentage': round(negative_pct, 2),
                'neutral_count': int(neutral_count),
                'neutral_percentage': round(neutral_pct, 2),
                'avg_sentiment_score': round(avg_sentiment_score, 3),
                'top_keywords': top_keywords,
                'keyword_string': ', '.join(top_keywords)
            }

            print(f"  Topic {topic} ({topic_size} reviews): {dominant_sentiment}")
            print(f"    Keywords: {', '.join(top_keywords[:3])}")
            print(f"    Sentiment: +{positive_count} ({positive_pct:.1f}%), "
                  f"-{negative_count} ({negative_pct:.1f}%), "
                  f"~{neutral_count} ({neutral_pct:.1f}%)")

        # ========================
        # 5. SAVE RESULTS
        # ========================
        print("\n4. Saving results...")

        # Save dataframe with topics and sentiment
        df.to_csv(f"{model_dir}/reviews_with_topics_sentiment.csv",
                  index=False, encoding="utf-8-sig")

        # Save topic-sentiment summary
        topic_summary_df = pd.DataFrame.from_dict(topic_sentiment_stats, orient='index')
        topic_summary_df.index.name = 'topic'
        topic_summary_df.reset_index(inplace=True)
        topic_summary_df.to_csv(f"{model_dir}/topic_sentiment_summary.csv",
                                index=False, encoding="utf-8-sig")

        # Save overall movie statistics (including sentence-level)
        overall_stats = {
            'movie_name': movie_name,
            'total_reviews': total_reviews,
            'total_sentences': total_sentences,
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'review_level_sentiment': {
                'positive': int(sentiment_counts.get('positive', 0)),
                'positive_percentage': round((sentiment_counts.get('positive', 0) / total_reviews) * 100, 2),
                'negative': int(sentiment_counts.get('negative', 0)),
                'negative_percentage': round((sentiment_counts.get('negative', 0) / total_reviews) * 100, 2),
                'neutral': int(sentiment_counts.get('neutral', 0)),
                'neutral_percentage': round((sentiment_counts.get('neutral', 0) / total_reviews) * 100, 2),
            },
            'sentence_level_sentiment': {
                'positive': int(sentence_counts.get('positive', 0)),
                'positive_percentage': round((sentence_counts.get('positive', 0) / total_sentences) * 100, 2),
                'negative': int(sentence_counts.get('negative', 0)),
                'negative_percentage': round((sentence_counts.get('negative', 0) / total_sentences) * 100, 2),
                'neutral': int(sentence_counts.get('neutral', 0)),
                'neutral_percentage': round((sentence_counts.get('neutral', 0) / total_sentences) * 100, 2),
            },
            'total_topics': len([t for t in set(topics) if t != -1]),
            'topic_sentiment_distribution': {}
        }

        # Add topic sentiment distribution
        for sentiment_type in ['POSITIVE', 'NEGATIVE', 'NEUTRAL', 'MIXED', 'MOSTLY_POSITIVE', 'MOSTLY_NEGATIVE']:
            count = len([t for t, stats in topic_sentiment_stats.items()
                         if stats['dominant_sentiment'] == sentiment_type])
            overall_stats['topic_sentiment_distribution'][sentiment_type] = count

        with open(f"{model_dir}/overall_stats.json", 'w', encoding='utf-8') as f:
            json.dump(overall_stats, f, indent=2)

        # Save BERTopic model
        topic_model.save(f"{model_dir}/topic_model")

        # ========================
        # 6. CREATE VISUALIZATIONS
        # ========================
        print("\n5. Creating visualizations...")

        try:
            # Create sentiment-topic visualization
            self._create_sentiment_topic_visualization(topic_summary_df, model_dir, movie_name)

            # Create sentence-level sentiment visualization
            self._create_sentence_sentiment_visualization(sentence_df, model_dir, movie_name)

            # Create BERTopic visualizations (if multiple topics)
            valid_topics = [t for t in set(topics) if t != -1]
            if len(valid_topics) > 1:
                # Topics overview
                fig = topic_model.visualize_topics()
                fig.write_html(f"{model_dir}/topics_overview.html")

                # Barchart
                fig = topic_model.visualize_barchart(n_words=10)
                fig.write_html(f"{model_dir}/topics_barchart.html")

                # Topics over time (if dates available)
                if dates and len(dates) == len(docs):
                    try:
                        topics_over_time = topic_model.topics_over_time(
                            docs=docs,
                            topics=topics,
                            timestamps=dates,
                            nr_bins=20
                        )
                        fig = topic_model.visualize_topics_over_time(topics_over_time)
                        fig.write_html(f"{model_dir}/topics_over_time.html")
                    except:
                        pass

                print("✓ Created topic visualizations")

            print("✓ Created sentiment-topic visualization")

        except Exception as e:
            print(f"✗ Visualization error: {str(e)}")

        print(f"\n✓ Analysis complete for {movie_name}")
        print(f"  Results saved to: {model_dir}")
        print(f"  Sentence-level results: {len(sentence_df)} sentences analyzed")
        print(f"  Review-level results: {len(df)} reviews with aggregated sentiment")

        return {
            'movie_name': movie_name,
            'overall_stats': overall_stats,
            'topic_sentiment_stats': topic_sentiment_stats,
            'sentence_results': sentence_df,
            'review_mapping': review_mapping_df
        }

    def _create_sentence_sentiment_visualization(self, sentence_df, model_dir, movie_name):
        """Create visualization of sentence-level sentiment"""
        try:
            import plotly.express as px
            import plotly.graph_objects as go

            # Create histogram of sentiment scores
            fig1 = px.histogram(
                sentence_df,
                x='sentiment_score',
                color='sentiment_label',
                nbins=30,
                title=f'Sentence-Level Sentiment Score Distribution - {movie_name}',
                labels={'sentiment_score': 'Sentiment Score', 'count': 'Number of Sentences'},
                color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
            )
            fig1.write_html(f"{model_dir}/sentence_sentiment_histogram.html")

            # Create pie chart of sentiment distribution
            sentiment_counts = sentence_df['sentiment_label'].value_counts()
            fig2 = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title=f'Sentence-Level Sentiment Distribution - {movie_name}',
                color=sentiment_counts.index,
                color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
            )
            fig2.write_html(f"{model_dir}/sentence_sentiment_pie.html")

            print("✓ Created sentence-level visualizations")

        except Exception as e:
            print(f"  Warning: Could not create sentence visualization - {str(e)}")

    def _create_sentiment_topic_visualization(self, topic_summary_df, model_dir, movie_name):
        """Create visualization of sentiment per topic"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px

            # Prepare data
            topic_summary_df = topic_summary_df.sort_values('topic')

            # Create grouped bar chart
            fig = go.Figure()

            # Add bars for each sentiment
            fig.add_trace(go.Bar(
                x=topic_summary_df['topic'],
                y=topic_summary_df['positive_percentage'],
                name='Positive %',
                marker_color='green',
                text=[f"{p:.1f}%" for p in topic_summary_df['positive_percentage']],
                textposition='outside'
            ))

            fig.add_trace(go.Bar(
                x=topic_summary_df['topic'],
                y=topic_summary_df['negative_percentage'],
                name='Negative %',
                marker_color='red',
                text=[f"{p:.1f}%" for p in topic_summary_df['negative_percentage']],
                textposition='outside'
            ))

            fig.add_trace(go.Bar(
                x=topic_summary_df['topic'],
                y=topic_summary_df['neutral_percentage'],
                name='Neutral %',
                marker_color='gray',
                text=[f"{p:.1f}%" for p in topic_summary_df['neutral_percentage']],
                textposition='outside'
            ))

            # Update layout
            fig.update_layout(
                title=f"Review-Level Sentiment Distribution by Topic - {movie_name}",
                xaxis_title="Topic Number",
                yaxis_title="Percentage (%)",
                barmode='stack',
                hovermode='x unified',
                height=600,
                showlegend=True
            )

            # Add keyword annotations
            annotations = []
            for i, row in topic_summary_df.iterrows():
                annotations.append(dict(
                    x=row['topic'],
                    y=105,
                    text=f"<b>Topic {row['topic']}</b><br>{row['keyword_string'][:60]}...",
                    showarrow=False,
                    font=dict(size=10),
                    xref="x",
                    yref="y"
                ))

            fig.update_layout(annotations=annotations)

            # Save
            fig.write_html(f"{model_dir}/sentiment_by_topic.html")

            # Also create a summary table visualization
            fig2 = go.Figure(data=[go.Table(
                header=dict(
                    values=['Topic', 'Size', 'Dominant Sentiment',
                            'Positive %', 'Negative %', 'Neutral %', 'Top Keywords'],
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[
                        topic_summary_df['topic'],
                        topic_summary_df['topic_size'],
                        topic_summary_df['dominant_sentiment'],
                        [f"{p:.1f}%" for p in topic_summary_df['positive_percentage']],
                        [f"{p:.1f}%" for p in topic_summary_df['negative_percentage']],
                        [f"{p:.1f}%" for p in topic_summary_df['neutral_percentage']],
                        topic_summary_df['keyword_string']
                    ],
                    fill_color='lavender',
                    align='left'
                )
            )])

            fig2.update_layout(
                title=f"Topic-Sentiment Summary - {movie_name}",
                height=400 + (len(topic_summary_df) * 20)
            )

            fig2.write_html(f"{model_dir}/topic_sentiment_summary_table.html")

        except Exception as e:
            print(f"  Warning: Could not create visualization - {str(e)}")


# ============================
# MAIN EXECUTION
# ============================
def main():
    """Main execution function"""
    print("=" * 70)
    print("MOVIE REVIEW TOPIC & SENTIMENT ANALYSIS")
    print("Mode: PER-SENTENCE SENTIMENT ANALYSIS")
    print("=" * 70)

    print("\nLoading data...")
    try:
        df = pd.read_csv("reviews.csv")
        print(f"✓ Loaded {len(df)} reviews")

        # Parse date
        df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
        df = df.dropna(subset=["date"])

        # Keep only needed columns
        df = df[["movie_name", "review_text", "date"]]

        # Filter only
        target_movie = "Hello Love Goodbye"
        df = df[df["movie_name"] == target_movie]

        if df.empty:
            print(f"✗ No reviews found for '{target_movie}'")
            return

        print(f"✓ Filtered to {len(df)} reviews for '{target_movie}'")

    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return

    # Initialize analyzer
    print("\nInitializing analyzer...")
    analyzer = MovieTopicSentimentAnalyzer()

    # Analyze only the Kingmaker movie
    result = analyzer.analyze_movie(target_movie, df)

    if result:
        print(f"\n{'=' * 60}")
        print("ANALYSIS COMPLETE")
        print("=" * 60)

        stats = result['overall_stats']
        print(f"Total Reviews: {stats['total_reviews']}")
        print(f"Total Sentences: {stats['total_sentences']}")
        print(f"Review-Level Positive: {stats['review_level_sentiment']['positive_percentage']:.1f}%")
        print(f"Sentence-Level Positive: {stats['sentence_level_sentiment']['positive_percentage']:.1f}%")
        print(f"Topics Found: {stats['total_topics']}")

        # Save a comprehensive summary
        summary_df = pd.DataFrame([{
            'Movie': target_movie,
            'Total Reviews': stats['total_reviews'],
            'Total Sentences': stats['total_sentences'],
            'Review Positive %': stats['review_level_sentiment']['positive_percentage'],
            'Review Negative %': stats['review_level_sentiment']['negative_percentage'],
            'Sentence Positive %': stats['sentence_level_sentiment']['positive_percentage'],
            'Sentence Negative %': stats['sentence_level_sentiment']['negative_percentage'],
            'Topics Found': stats['total_topics']
        }])

        summary_df.to_csv(f"./Models/{target_movie.replace(' ', '_')}/comprehensive_summary.csv",
                          index=False, encoding="utf-8-sig")

        print(f"\n✓ All results saved to: ./Models/{target_movie.replace(' ', '_')}/")
    else:
        print("✗ Analysis failed for Kingmaker")


if __name__ == "__main__":
    main()