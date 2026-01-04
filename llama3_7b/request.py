import requests
import json
import urllib.parse


class ModelClient:
    def __init__(self, server_url, username="your_username", password="your_password"):
        self.url = f"http://{server_url}:8000/generate"
        #encoded_password = urllib.parse.quote(password)
        self.auth = (username, password)
        
    
    def generate(self, prompt, max_length=512, temperature=1):
        data = {
            "prompt": prompt,
            "max_length": max_length,
            "temperature": temperature
        }
        
        response = requests.post(self.url, json=data, auth=self.auth)

        return response.json()

if __name__ == "__main__":
    client = ModelClient("44.221.203.57", username='UStAilaN', password='pK9#mJ4$xL2@')  # Replace with actual IP
    
    # Test the service
    result = client.generate("Once upon a time")
    print(f"Generated text: {result['generated_text']}")

    print('\n\n')
    result = client.generate("Hello, how are you today?")
    print(f"Generated text: {result['generated_text']}")


    print('\n\n')
    result = client.generate("The quick brown fox")
    print(f"Generated text: {result['generated_text']}")
