import runpod
import json
from .models.sentiment_bert import predict_sentiment
from .models.mental_health_bert import classify_mental_health
from .models.llama_counsel import generate_response

def handler(event):
    """
    This is the handler function that will be called by RunPod.
    """
    try:
        # Get the input from the event
        input_data = event["input"]
        prompt = input_data.get("prompt")
        
        if not prompt:
            return {
                "error": "No prompt provided"
            }

        # Process the prompt with all models
        sentiment_result = predict_sentiment(prompt)
        mental_health_result = classify_mental_health(prompt)
        llama_response = generate_response(prompt)

        # Return the results
        return {
            "response": llama_response,
            "sentiment": sentiment_result,
            "mental_health": mental_health_result
        }

    except Exception as e:
        return {
            "error": str(e)
        }

# Start the RunPod handler
runpod.serverless.start({"handler": handler}) 