from google import genai

client = genai.Client(api_key="YOUR_API_KEY")

for batch in client.batches.list(config={"page_size": 100}):
    print("name:", batch.name)
    print("state:", batch.state)
    print("create_time:", batch.create_time)
    print("start_time:", getattr(batch, "start_time", None))
    print("end_time:", getattr(batch, "end_time", None))
    print("error:", getattr(batch, "error", None))
    print("-" * 80)