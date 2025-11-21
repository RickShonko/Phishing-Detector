# 🚀 Backend Developer Quick Start Guide

## Your Mission
Build and integrate the AI phishing detection system into the backend API.

---

## 📁 Project Structure You Need to Create

```
phishing-detector/
├── dataset/
│   └── kenyan_phishing_data.json          # ✅ Already created for you
├── model/
│   ├── train.py                            # ✅ Already created for you
│   ├── predict.py                          # ✅ Already created for you
│   ├── requirements.txt                    # Create this
│   └── saved_model/                        # Created after training
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer files...
│       └── label_map.json
├── backend/
│   ├── main.py                             # ✅ Already created for you
│   └── requirements.txt                    # ✅ Already created for you
└── README.md
```

---

## 🎯 Step-by-Step Implementation

### STEP 1: Initial Setup (5 minutes)

```bash
# Clone your repo or create project directory
mkdir ai-phishing-detector-ke
cd phishing-detector

# Create folder structure
mkdir dataset model backend

# Create virtual environment (HIGHLY RECOMMENDED)
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### STEP 2: Setup Dataset (2 minutes)

```bash
cd dataset
# Copy the kenyan_phishing_data.json file I created into this folder
# You can add more examples to improve accuracy
```

**💡 Pro Tip:** The dataset I created has 30 examples. For better accuracy, try to add 20-50 more real Kenyan phishing examples you've seen or can find online.

### STEP 3: Install Model Dependencies (5 minutes)

```bash
cd ../backend
pip install -r requirements.txt

# If you have a GPU (much faster):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only (slower but works):
pip install torch torchvision torchaudio
```

### STEP 4: Train the Model (20-60 minutes)

```bash
# Make sure you're in the model directory
python train.py
```

**What to expect:**
- On CPU: 30-60 minutes
- On GPU: 5-15 minutes
- On Google Colab (free GPU): 10 minutes

**⚡ HACKATHON TIP:** If your laptop is slow, use **Google Colab**:
1. Upload `train.py` and `kenyan_phishing_data.json` to Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Run the training
4. Download the `saved_model` folder
5. Put it in your local `model/` directory

**Output:** You'll see something like:
```
✨ Training Complete!
📊 Final Results:
   - Accuracy: 0.9333
   - F1 Score: 0.9287
   - Precision: 0.9412
   - Recall: 0.9333
💾 Saving model to ./saved_model
✅ Model saved successfully!
```

### STEP 5: Test the Model (5 minutes)

Create a test script `test_model.py`:

```python
from predict import predict_phishing

# Test with a phishing message
test_message = "URGENT: Your KRA PIN suspended. Click http://fake-kra.com to verify now!"
result = predict_phishing(test_message)

print("Classification:", result['classification'])
print("Explanation:", result['explanation'])
print("Confidence:", result['confidence_percentage'])
```

Run it:
```bash
python test_model.py
```

### STEP 6: Setup Backend API (5 minutes)

```bash
cd ../backend
pip install -r requirements.txt
```

### STEP 7: Run the API (2 minutes)

```bash
# Make sure you're in the backend directory
uvicorn main:app --reload --port 8000
```

**You should see:**
```
🚀 Starting Kenyan Phishing Detector API
📡 API available at http://localhost:8000
📚 Documentation at http://localhost:8000/docs
```

### STEP 8: Test the API (5 minutes)

**Option 1: Use the built-in docs**
- Open browser: http://localhost:8000/docs
- Click on "POST /analyze-message"
- Click "Try it out"
- Paste a test message
- Click "Execute"

**Option 2: Use curl**
```bash
curl -X POST "http://localhost:8000/analyze-message" \
  -H "Content-Type: application/json" \
  -d '{"message": "Congratulations! Send your MPesa PIN to claim Ksh 50,000 prize!"}'
```

**Option 3: Use Python**
```python
import requests

response = requests.post(
    "http://localhost:8000/analyze-message",
    json={"message": "Your KCB account suspended. Verify at http://fake-kcb.com"}
)
print(response.json())
```

---

## 🔧 Troubleshooting

### Problem: "Module not found: transformers"
**Solution:** Make sure you activated your venv and ran `pip install -r requirements.txt`

### Problem: Training is too slow
**Solution:** Use Google Colab with GPU (free) or reduce epochs in `train.py` from 3 to 2

### Problem: Model accuracy is low
**Solution:** Add more diverse examples to your dataset (aim for 50-100 examples minimum)

### Problem: API returns "DEMO MODE"
**Solution:** The model isn't loaded. Check that `model/saved_model/` exists and contains model files

### Problem: CORS errors in frontend
**Solution:** Already handled in `main.py` with `allow_origins=["*"]`

---

## 🎨 Integration with Frontend

Your frontend developer needs to make POST requests to:

```javascript
// Example fetch request
fetch('http://localhost:8000/analyze-message', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    message: userInputMessage
  })
})
.then(response => response.json())
.then(data => {
  console.log('Risk:', data.classification);
  console.log('Explanation:', data.explanation);
  console.log('Action:', data.recommended_action);
});
```

Give them:
- **API URL:** `http://localhost:8000` (or your deployed URL)
- **Endpoint:** `POST /analyze-message`
- **Request format:** `{"message": "text here"}`
- **API docs:** `http://localhost:8000/docs`

---

## 📊 API Endpoints Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Health check |
| `/health` | GET | Detailed health status |
| `/analyze-message` | POST | Analyze single message |
| `/batch-analyze` | POST | Analyze multiple messages |
| `/stats` | GET | Get usage statistics |
| `/docs` | GET | Interactive API documentation |

---

## 🚀 Deployment (After Hackathon Demo)

For the hackathon, running locally is fine. For deployment:

1. **Heroku** (Easiest)
2. **Railway** (Great for hackathons)
3. **Hugging Face Spaces** (Built for ML models)
4. **Google Cloud Run** (Professional)

I can help you deploy when ready!

---

## 💡 Hackathon Demo Tips

1. **Have backup examples ready:** Prepare 5-10 test messages (real phishing, fake safe messages)
2. **Show the progression:** Raw message → API call → JSON response → Beautiful UI
3. **Highlight Kenyan context:** Use MPesa, KRA, HELB examples
4. **Show confidence scores:** Judges love seeing the probability percentages
5. **Demo the API docs:** Show http://localhost:8000/docs - judges love interactive docs

---

## ⏱️ Time Breakdown

- Setup & Installation: **20 minutes**
- Model Training: **30 minutes** (or 10 min on Colab)
- Testing & Integration: **20 minutes**
- **Total:** ~70 minutes to have everything working!

---

## 🆘 Need Help?

If you get stuck:
1. Check the error message carefully
2. Ensure all files are in the right directories
3. Verify your virtual environment is activated
4. Check that the model trained successfully
5. Ask me for help! 😊

---

## ✅ Pre-Demo Checklist

- [ ] Dataset created with 30+ examples
- [ ] Model trained successfully (accuracy > 85%)
- [ ] API running without errors
- [ ] Tested with sample messages
- [ ] API docs accessible at /docs
- [ ] Frontend can connect to backend
- [ ] Have demo script ready
- [ ] Battery charged, internet stable 😄

---

**You've got this! Let's build something amazing for Kenya! 🇰🇪🚀**