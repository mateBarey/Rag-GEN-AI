import weaviate

class RAGModelTester:
    def __init__(self, weaviate_client, model_name):
        self.weaviate_client = weaviate_client
        self.model_name = model_name

    def test_chunk_presence(self, test_chunks):
        results = {}
        for chunk in test_chunks:
            result = self._search_chunk(chunk)
            results[chunk] = result
        return results

    def _search_chunk(self, chunk_text):
        response = self.weaviate_client.query.get(self.model_name, ["content"]).with_near_text({"concepts": [chunk_text]}).with_limit(1).do()
        if response.get('data') and response['data']['Get'][self.model_name]:
            return True
        return False

# Example usage of the RAGModelTester

# Initialize Weaviate client
connection_params = {
    "url": "http://localhost:8080",
}
weaviate_client = weaviate.Client(
    url=connection_params['url'],
    timeout_config=(30, 30)
)

# Test chunks from each PDF
test_chunks = [
    "Texas Business Law Example Text",  # Replace with actual text from the PDF
    "Black's Law Dictionary Example Text",  # Replace with actual text from the PDF
    "Real Estate Law Example Text",  # Replace with actual text from the PDF
    "IRS Code Example Text",  # Replace with actual text from the PDF
    "Tax Liens Investing Example Text"  # Replace with actual text from the PDF
]

# Create and use the tester
tester = RAGModelTester(weaviate_client, "AggregateModel")
test_results = tester.test_chunk_presence(test_chunks)

for chunk, result in test_results.items():
    print(f"Chunk: '{chunk}' Presence: {result}")
