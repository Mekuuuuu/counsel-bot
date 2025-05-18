from huggingface_hub import InferenceClient
import json

repo_id = "Mekuu/LLAMA3.1-8b-Counsel-v1.3"

# llm_client = InferenceClient(
#     model=repo_id,
#     timeout=120
# )

llm_client = InferenceClient(
    provider="auto",
    api_key="HF-TOKEN",
)

# def call_llm(prompt: str):
#     response = llm_client.chat_completion(
#         messages=[
#             {"role": "user", "content": prompt}
#         ],
#         max_tokens=200,
#         temperature=0.7
#     )
#     return response.choices[0].message["content"]

response = llm_client.chat.completions.create(
    model="Mekuu/LLAMA3.1-8b-Counsel-v1.3",
    messages=[
        {
            "role": "user",
            "content": "Who are you?"
        }
    ],
)
print(response)