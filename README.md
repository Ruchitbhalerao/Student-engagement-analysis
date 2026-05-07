# Student Engagement Analyzer
## Hybrid ImLN-BiLSTM Framework

### Setup
1. Install dependencies:
   pip install -r requirements.txt

2. Download face landmarker model (auto-downloads on first run)

3. Run the app:
   streamlit run app.py

### Files
- best_model.pt       : Trained model weights (INPUT_DIM=88)
- scaler.pkl          : Feature scaler
- app.py              : Streamlit web app
- requirements.txt    : Python dependencies

### Model Architecture
- 2x Bidirectional LSTM + 2x Bidirectional GRU
- Temporal self-attention per branch
- Adaptive feature gate
- Attention-gated ensemble fusion
- Focal MSE loss (gamma=2.0)

### Performance (DAiSEE)
- Baseline (Tian et al. 2024) : MSE = 0.0386
- Our model                   : MSE = 0.0383