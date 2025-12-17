@echo off
echo ========================================
echo MOVIE SENTIMENT ANALYSIS
echo ========================================
echo.

echo 1. Installing required packages...
pip install transformers pandas numpy tqdm matplotlib seaborn torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo 2. Running quick test...
python test_analysis.py

echo.
echo 3. Running full analysis...
python analyze_movie_sentiment.py

echo.
echo ========================================
echo ANALYSIS COMPLETE!
echo ========================================
echo.
pause