"""
Kenyan Phishing Detector - FastAPI Backend
Main API server for phishing detection
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import sys
import os

# Add model directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import prediction module
try:
    from model.predict import predict_phishing
except ImportError:
    # Fallback for development
    def predict_phishing(message: str):
        return {
            "risk_score": 0.85,
            "classification": "High Risk (Phishing)",
            "explanation": "DEMO MODE: Model not loaded. This is a sample response.",
            "recommended_action": "Please train and load the model first.",
            "detected_patterns": ["demo_pattern"],
            "confidence_percentage": "85.0%"
        }

# Initialize FastAPI app
app = FastAPI(
    title="🇰🇪 Kenyan Phishing Detector API",
    description="AI-powered phishing detection system for Kenyan SMS, WhatsApp, and email messages",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class MessageRequest(BaseModel):
    """Request model for message analysis"""
    message: str = Field(
        ..., 
        min_length=1,
        max_length=5000,
        description="The SMS, WhatsApp, or email message to analyze"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "URGENT: Your KRA PIN has been suspended. Click http://kra-fake.com to reactivate immediately."
            }
        }

class AnalysisResult(BaseModel):
    """Response model for analysis results"""
    risk_score: float = Field(..., description="Confidence score (0-1)")
    classification: str = Field(..., description="Risk classification")
    explanation: str = Field(..., description="Human-readable explanation")
    recommended_action: str = Field(..., description="What the user should do")
    detected_patterns: List[str] = Field(..., description="Suspicious patterns found")
    confidence_percentage: str = Field(..., description="Confidence as percentage")

class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    message: str
    version: str

# Statistics storage (in production, use a proper database)
analysis_stats = {
    "total_analyzed": 0,
    "phishing_detected": 0,
    "suspicious_detected": 0,
    "legitimate_detected": 0
}

# API Endpoints

@app.get("/", response_model=HealthCheck)
async def root():
    """Root endpoint - API health check"""
    return HealthCheck(
        status="online",
        message="🇰🇪 Kenyan Phishing Detector API is running",
        version="1.0.0"
    )

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        message="All systems operational",
        version="1.0.0"
    )

@app.post("/analyze-message", response_model=AnalysisResult, status_code=status.HTTP_200_OK)
async def analyze_message(request: MessageRequest):
    """
    Main endpoint: Analyze a message for phishing indicators
    
    **Parameters:**
    - message: The text message to analyze
    
    **Returns:**
    - Complete analysis with risk score, classification, and recommendations
    """
    try:
        # Get the message
        message = request.message.strip()
        
        if not message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message cannot be empty"
            )
        
        # Analyze the message
        result = predict_phishing(message)
        
        # Update statistics
        analysis_stats["total_analyzed"] += 1
        if "High Risk" in result["classification"]:
            analysis_stats["phishing_detected"] += 1
        elif "Medium Risk" in result["classification"]:
            analysis_stats["suspicious_detected"] += 1
        else:
            analysis_stats["legitimate_detected"] += 1
        
        return AnalysisResult(**result)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/stats")
async def get_statistics():
    """Get API usage statistics"""
    return {
        "statistics": analysis_stats,
        "detection_rate": {
            "phishing_rate": f"{(analysis_stats['phishing_detected'] / max(analysis_stats['total_analyzed'], 1)) * 100:.1f}%",
            "suspicious_rate": f"{(analysis_stats['suspicious_detected'] / max(analysis_stats['total_analyzed'], 1)) * 100:.1f}%",
            "legitimate_rate": f"{(analysis_stats['legitimate_detected'] / max(analysis_stats['total_analyzed'], 1)) * 100:.1f}%"
        }
    }

@app.post("/batch-analyze")
async def batch_analyze(messages: List[str]):
    """
    Analyze multiple messages at once
    Useful for processing bulk messages
    """
    if len(messages) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 100 messages per batch"
        )
    
    results = []
    for msg in messages:
        try:
            result = predict_phishing(msg)
            results.append({
                "message": msg[:100] + "..." if len(msg) > 100 else msg,
                "analysis": result
            })
        except Exception as e:
            results.append({
                "message": msg[:100] + "..." if len(msg) > 100 else msg,
                "error": str(e)
            })
    
    return {"batch_results": results, "total_processed": len(results)}

# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("🚀 Starting Kenyan Phishing Detector API")
    print("📡 API available at http://localhost:8000")
    print("📚 Documentation at http://localhost:8000/docs")
    print("🇰🇪 Protecting Kenya, one message at a time!")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    print("👋 Shutting down Kenyan Phishing Detector API")
    print(f"📊 Total messages analyzed: {analysis_stats['total_analyzed']}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )