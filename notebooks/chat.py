from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5.6",
    input="Explain Apache Spark in simple terms."
)

print(response.output_text)