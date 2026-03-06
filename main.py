import requests
from config import API_KEY

def classify_text(text):
    api_url="https://router.huggingface.co/hf-inference/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    headers={"Authorization": f"bearer {API_KEY}"}
    payload={"inputs":text} 
    response=requests.post(api_url,headers=headers,json=payload)
    try:
        response.raise_for_status() 
        return response.json()
    except ValueError:
        print(f"Invalid JSON response: {response.text}")
    except requests.RequestsExceptions as e:
        print(f"Request error::{e}")
        return{}
def print_result(data):
    if not data:
        return
    try:
        if isinstance(data,list):
            if data and isinstance(data[0],list):
                result=data[0][0]
            else:
                result=data[0]
        else:
            result=data

        
        label=result.get("label","UNKNOWN")
        score=result.get("score",0)
        percentage=score*100
        emoji="😁"  if label=="POSITIVE" else "😢"
        icon="✅" if label=="POSITIVE" else "❌"
        print(f"Sentiment: {icon} {label} {emoji}")
        print(f"Confidence: {percentage:.2f}%")
        print("-" * 30)
    except (KeyError, IndexError, TypeError) as e:
        print(f"[!] Could not parse response: {data} ({e})")
def main():
    print("==== AI Sentiment Analyzer (Type 'quit' to exit) ====")
    while True:
        user_input = input("\nEnter text: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            print("Exiting...")
            break
        if not user_input:
            continue
        print("Analyzing...")
        data = classify_text(user_input)
        print_result(data)
if __name__ == "__main__":
    main()