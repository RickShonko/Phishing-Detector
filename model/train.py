"""
Kenyan Phishing Detector - Model Training Script
Fine-tunes a transformer model for phishing detection
"""

import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Configuration
MODEL_NAME = "distilbert-base-uncased"  # Fast and efficient for hackathon
OUTPUT_DIR = "./saved_model"
DATASET_PATH = "../dataset/kenyan_phishing_data.json"

# Label mapping
LABEL_MAP = {
    "Phishing": 0,
    "Suspicious": 1,
    "Legitimate": 2
}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def load_dataset(dataset_path):
    """Load and prepare the dataset"""
    print(f"📂 Loading dataset from {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract texts and labels
    texts = [item['text'] for item in data]
    labels = [LABEL_MAP[item['label']] for item in data]
    explanations = [item['explanation'] for item in data]
    
    print(f"✅ Loaded {len(texts)} examples")
    print(f"   - Phishing: {labels.count(0)}")
    print(f"   - Suspicious: {labels.count(1)}")
    print(f"   - Legitimate: {labels.count(2)}")
    
    return texts, labels, explanations

def prepare_datasets(texts, labels, test_size=0.2):
    """Split and prepare datasets for training"""
    print(f"\n📊 Splitting dataset (train: {1-test_size}, test: {test_size})")
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    train_dataset = Dataset.from_dict({
        'text': train_texts,
        'label': train_labels
    })
    
    test_dataset = Dataset.from_dict({
        'text': test_texts,
        'label': test_labels
    })
    
    return train_dataset, test_dataset

def tokenize_data(dataset, tokenizer):
    """Tokenize the dataset"""
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=256
        )
    
    tokenized = dataset.map(tokenize_function, batched=True)
    return tokenized

def compute_metrics(pred):
    """Compute evaluation metrics"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model():
    """Main training function"""
    print("🚀 Starting Kenyan Phishing Detector Training\n")
    
    # Load dataset
    texts, labels, explanations = load_dataset(DATASET_PATH)
    
    # Prepare datasets
    train_dataset, test_dataset = prepare_datasets(texts, labels)
    
    # Load tokenizer and model
    print(f"\n🤖 Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=REVERSE_LABEL_MAP,
        label2id=LABEL_MAP
    )
    
    # Tokenize datasets
    print("🔤 Tokenizing data...")
    train_tokenized = tokenize_data(train_dataset, tokenizer)
    test_tokenized = tokenize_data(test_dataset, tokenizer)
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        logging_steps=10,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\n🏋️ Training model...")
    trainer.train()
    
    # Evaluate
    print("\n📈 Evaluating model...")
    eval_results = trainer.evaluate()
    
    print("\n✨ Training Complete!")
    print(f"📊 Final Results:")
    print(f"   - Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"   - F1 Score: {eval_results['eval_f1']:.4f}")
    print(f"   - Precision: {eval_results['eval_precision']:.4f}")
    print(f"   - Recall: {eval_results['eval_recall']:.4f}")
    
    # Save final model
    print(f"\n💾 Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save label mapping
    with open(f"{OUTPUT_DIR}/label_map.json", 'w') as f:
        json.dump(LABEL_MAP, f)
    
    print("✅ Model saved successfully!")
    return trainer, eval_results

if __name__ == "__main__":
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Using device: {device}")
    
    if device == "cpu":
        print("⚠️  Warning: Training on CPU will be slow. Consider using Google Colab with GPU.")
    
    # Train
    trainer, results = train_model()
    
    print("\n🎉 All done! Your model is ready for deployment.")