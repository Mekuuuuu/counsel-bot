from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import torch
import os
from dotenv import load_dotenv
from models.sentiment_bert import predict_sentiment
from models.mental_health_bert import classify_mental_health
from models.llama_counsel import generate_response

# Load environment variables
load_dotenv()

app = FastAPI()

class TextRequest(BaseModel):
    prompt: str

class AnalysisResponse(BaseModel):
    response: str
    sentiment: Dict[str, Any]
    mental_health: Dict[str, Any]

@app.post("/generate", response_model=AnalysisResponse)
async def process_text(request: TextRequest) -> AnalysisResponse:
    try:
        # Get sentiment analysis
        sentiment_result = predict_sentiment(request.prompt)
        
        # Get mental health classification
        mental_health_result = classify_mental_health(request.prompt)
        
        # Generate response using Llama
        llama_response = generate_response(request.prompt)
        
        return AnalysisResponse(
            response=llama_response,
            sentiment=sentiment_result,
            mental_health=mental_health_result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000"))
    ) 