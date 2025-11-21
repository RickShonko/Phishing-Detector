"""
Kenyan Phishing Detector - Prediction Module
Handles inference and explanation generation
"""

import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Tuple
import re

# Model paths
MODEL_PATH = "./model/saved_model"

class PhishingDetector:
    """Main phishing detection class"""
    
    def __init__(self, model_path: str = MODEL_PATH):
        """Initialize the detector with trained model"""
        print(f"🔄 Loading model from {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        
        # Load label mapping
        with open(f"{model_path}/label_map.json", 'r') as f:
            self.label_map = json.load(f)
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        # Kenyan-specific keywords for enhanced detection
        self.kenyan_institutions = [
            'safaricom', 'mpesa', 'kra', 'helb', 'kcb', 'equity', 
            'ncba', 'airtel', 'hustler fund', 'ntsa', 'nhif', 'nssf',
            'kenya power', 'kplc', 'nairobi', 'mombasa'
        ]
        
        self.danger_patterns = [
            r'send.*pin', r'confirm.*pin', r'reply.*pin',
            r'click here', r'urgent', r'suspended', r'expires',
            r'won.*ksh', r'claim.*prize', r'verify.*account'
        ]
        
        print("✅ Model loaded successfully")
    
    def extract_suspicious_patterns(self, text: str) -> list:
        """Extract suspicious patterns from text"""
        text_lower = text.lower()
        found_patterns = []
        
        # Check for danger patterns
        for pattern in self.danger_patterns:
            if re.search(pattern, text_lower):
                found_patterns.append(pattern.replace('.*', ' '))
        
        # Check for suspicious URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        if urls:
            for url in urls:
                if not any(official in url.lower() for official in ['safaricom.co.ke', 'kra.go.ke', 'itax.kra.go.ke', 'kcb.co.ke', 'equity.co.ke']):
                    found_patterns.append(f"suspicious_url: {url}")
        
        # Check for institution impersonation
        for institution in self.kenyan_institutions:
            if institution in text_lower:
                found_patterns.append(f"mentions: {institution}")
        
        return found_patterns
    
    def predict(self, text: str) -> Tuple[str, float, int]:
        """
        Predict if text is phishing
        Returns: (label, confidence, label_id)
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        label = self.reverse_label_map[predicted_class]
        return label, confidence, predicted_class
    
    def generate_explanation(self, text: str, label: str, patterns: list) -> str:
        """Generate human-readable explanation"""
        explanations = []
        text_lower = text.lower()
        
        if label == "Phishing":
            # High risk explanations
            if any('pin' in p for p in patterns):
                explanations.append("🚨 Requests sensitive MPesa PIN or security credentials")
            
            if any('urgent' in text_lower for _ in range(1)):
                explanations.append("⚠️ Creates artificial urgency to pressure immediate action")
            
            if any('click' in p or 'suspicious_url' in p for p in patterns):
                explanations.append("🔗 Contains suspicious or unofficial links")
            
            if any(inst in text_lower for inst in self.kenyan_institutions):
                explanations.append("🏛️ Appears to impersonate a trusted Kenyan institution")
            
            if any(word in text_lower for word in ['won', 'prize', 'congratulations']):
                explanations.append("🎁 Offers unrealistic prizes or rewards")
            
            if not explanations:
                explanations.append("⚠️ Multiple phishing indicators detected in message content")
        
        elif label == "Suspicious":
            explanations.append("⚠️ Shows some red flags but not definitively phishing")
            
            if 'http' in text_lower:
                explanations.append("🔗 Contains links that should be verified")
            
            if any(word in text_lower for word in ['offer', 'limited', 'hurry']):
                explanations.append("⏰ Uses pressure tactics or limited-time offers")
        
        else:  # Legitimate
            explanations.append("✅ Appears to be from a legitimate source")
            
            if any(official in text_lower for official in ['safaricom.co.ke', 'kra.go.ke', '.go.ke']):
                explanations.append("🏛️ Uses official government or institutional domains")
        
        return " | ".join(explanations) if explanations else "Analysis based on message content and patterns."
    
    def get_recommended_action(self, label: str) -> str:
        """Get recommended action based on classification"""
        actions = {
            "Phishing": "🛑 DO NOT respond, click links, or share any information. Delete this message immediately. If it claims to be from a bank/institution, contact them directly using official channels.",
            "Suspicious": "⚠️ Be cautious. Do not click links or share sensitive information. Verify the sender through official channels (call the institution directly using numbers from their official website).",
            "Legitimate": "✅ This message appears safe, but always verify sender authenticity for sensitive transactions. When in doubt, contact the institution directly."
        }
        return actions.get(label, "Please verify the sender before taking any action.")
    
    def analyze_message(self, message: str) -> Dict:
        """
        Main analysis function - returns complete analysis
        """
        # Get prediction
        label, confidence, label_id = self.predict(message)
        
        # Extract patterns
        patterns = self.extract_suspicious_patterns(message)
        
        # Generate explanation
        explanation = self.generate_explanation(message, label, patterns)
        
        # Get recommended action
        action = self.get_recommended_action(label)
        
        # Map to risk levels
        risk_mapping = {
            "Phishing": "High Risk (Phishing)",
            "Suspicious": "Medium Risk (Suspicious)",
            "Legitimate": "Low Risk (Safe)"
        }
        
        return {
            "risk_score": confidence,
            "classification": risk_mapping[label],
            "explanation": explanation,
            "recommended_action": action,
            "detected_patterns": patterns[:3],  # Top 3 patterns
            "confidence_percentage": f"{confidence * 100:.1f}%"
        }

# Global instance
_detector = None

def get_detector():
    """Get or create detector instance"""
    global _detector
    if _detector is None:
        _detector = PhishingDetector()
    return _detector

def predict_phishing(message: str) -> Dict:
    """
    Main function to be called by the API
    """
    detector = get_detector()
    return detector.analyze_message(message)