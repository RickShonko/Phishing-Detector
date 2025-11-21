flowchart TD

%% User Layer
A[User Input\n(Email, SMS, URL, Text)] --> B[Frontend UI\n(Streamlit / Web App)]

%% Backend API
B --> C[FastAPI Backend API]

%% Preprocessing
C --> D[Preprocessing\n- Clean text\n- Tokenization\n- URL extraction]

%% Feature Extraction
D --> E[Feature Extraction\n(TF-IDF, URL Features, Patterns)]

%% ML Model
E --> F[ML Model\n(SVM / Logistic Regression / Naive Bayes)]

%% Output
F --> G[Prediction\n- Phishing / Legit\n- Risk Score]

%% Explainability
F --> H[Explainability Layer (LIME / SHAP)\nHighlights Suspicious Words]

%% Deployment Layer
G --> I[Deployment\n(HuggingFace / Render / Docker)]
H --> I
