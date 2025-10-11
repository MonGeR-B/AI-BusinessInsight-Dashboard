import openai
import os

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello"}]
    )
    print("API key works! Response:", response.choices[0].message.content)
except Exception as e:
    if "authentication" in str(e).lower() or "api key" in str(e).lower():
        print("Authentication failed: Invalid API key")
    else:
        print("Error:", e)
