from google import genai

client = genai.Client(api_key="AIzaSyDOXQxTdJDUFurdtHk3deRNmfVEi1W5sKw")

for m in client.models.list():
    print(m.name)