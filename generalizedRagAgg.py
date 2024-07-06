import weaviate
import fitz  # PyMuPDF
from typing import List
import dspy
import concurrent.futures
from textblob import TextBlob
import re 


class WeaviateRM:
    def __init__(self, class_name, weaviate_client, k=10):
        self.class_name = class_name
        self.client = weaviate_client
        self.k = k

    def retrieve(self, query, top_k=None):
        if top_k is None:
            top_k = self.k
        response = self.client.query.get(self.class_name, ["content"]).with_near_text({"concepts": [query]}).with_limit(top_k).do()
        return response["data"]["Get"][self.class_name]
    
class GeneralizedRAG:
    def __init__(self, model_name: str, model_input: str, pdf_source_files: List[str] = None):
        self.model_name = model_name
        self.model_input = model_input
        self.pdf_source_files = pdf_source_files if pdf_source_files else []
        
        # Initialize Weaviate client
        self.connection_params = {
            "url": "http://localhost:8080",
        }
        self.weaviate_client = weaviate.Client(
            url=self.connection_params['url'],
            timeout_config=(30, 30)
        )
        
        # Create schema
        self.schema = {
            "class": model_name,
            "vectorizer": "text2vec-contextionary",
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                },
                {
                    "name": "chunk_id",
                    "dataType": ["string"],
                }
            ]
        }
        self._create_schema()

        # Process PDFs and load data
        if self.pdf_source_files:
            self._process_pdfs()

        # Setup Retriever Model and RAG system
        self.retriever_model = WeaviateRM(model_name, weaviate_client=self.weaviate_client, k=10)
        self._setup_dspy()

    def _create_schema(self):
        try:
            self.weaviate_client.schema.create_class(self.schema)
            print(f"{self.model_name} class created successfully.")
        except weaviate.exceptions.UnexpectedStatusCodeException as e:
            if "already exists" in str(e):
                print(f"{self.model_name} class already exists in Weaviate.")
            else:
                print(f"Failed to create class: {e}")

    def _load_chunk_to_weaviate(self, chunk, chunk_id):
        self.weaviate_client.data_object.create(
            {
                "content": chunk,
                "chunk_id": str(chunk_id)
            },
            self.model_name
        )
    def _process_pdfs(self):
        all_chunks = []
        for pdf_path in self.pdf_source_files:
            text = self._extract_text_from_pdf(pdf_path)
            chunks = self._chunk_text(text)
            all_chunks.extend(chunks)
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self._load_chunk_to_weaviate, chunk, i) 
                       for i, chunk in enumerate(all_chunks)]
            concurrent.futures.wait(futures)

    def _extract_text_from_pdf(self, pdf_path):
        document = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
        return text

    def _chunk_text(self, text, chunk_size=1000):
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
        return chunks



    def _setup_dspy(self):
        self.llm = dspy.OllamaLocal(model=self.model_input)
        dspy.settings.configure(lm=self.llm, rm=self.retriever_model)
        
        class CustomSignature(dspy.Signature):
            """You are an expert. Answer the question based on the given context."""
            context: List[str] = dspy.InputField()
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()
        
        class CustomRAG(dspy.Module):
            def __init__(self, retriever_model):
                super().__init__()
                self.retriever_model = retriever_model
                self.generate = dspy.ChainOfThought(CustomSignature)
            
            def forward(self, question):
                context_results = self.retriever_model.retrieve(question, top_k=5)
                context = [result['content'] for result in context_results]
                answer_output = self.generate(context=context, question=question)
                detailed_answer = self.add_technical_details(answer_output.answer, context)
                final_answer = self.reflect_and_adjust(detailed_answer, context, question)
                return final_answer
            
            def add_technical_details(self, answer, context):
                detailed_answer = f"Technical details on the topic:\n{answer}\n"
                for additional_context in context:
                    detailed_answer += f"\nAdditional Context: {additional_context}\n"
                return detailed_answer
            
            def reflect_and_adjust(self, answer, context, question):
                relevance_score = self.evaluate_response(answer, context)
                if relevance_score < 0.75:
                    print(f"Adjusting response for query '{question}' due to low relevance score.")
                    answer = self.improve_response(answer, context, question)
                return answer
            
            def evaluate_response(self, response: str, context: List[str]) -> float:
                # Custom evaluation logic to score the relevance and accuracy of the response
                keyword_score = self.keyword_matching_score(response, context)
                sentiment_score = self.sentiment_analysis_score(response)
                
                # Weighted average of different scores (weights can be adjusted)
                overall_score = 0.6 * keyword_score + 0.4 * sentiment_score
                return overall_score

            def keyword_matching_score(self, response: str, context: List[str]) -> float:
                # Evaluate based on keyword matching
                keywords = set(re.findall(r'\b\w+\b', ' '.join(context)))
                response_keywords = set(re.findall(r'\b\w+\b', response))
                matched_keywords = keywords.intersection(response_keywords)
                
                if not keywords:
                    return 0
                
                return len(matched_keywords) / len(keywords)

            def sentiment_analysis_score(self, response: str) -> float:
                # Evaluate based on sentiment analysis
                analysis = TextBlob(response)
                # Assuming that neutral to positive sentiment is desired for technical accuracy
                return analysis.sentiment.polarity  # Scale between -1 (negative) to 1 (positive)

            
            def improve_response(self, response, context, question):
                improved_response = f"Improved technical answer to the query '{question}':\n{response}\n"
                for additional_context in context:
                    improved_response += f"\nFurther Context: {additional_context}\n"
                return improved_response
        
        self.custom_rag = CustomRAG(retriever_model=self.retriever_model)
    
    def ask_question(self, question: str):
        return self.custom_rag(question=question)

'''
# Example usage with multiple PDFs

pdf_paths = [
    "path/to/texas_bussiness_law.pdf",
    "path/to/blacksLawdictionary.pdf",
    "path/to/real_estatelaw.pdf",
    "path/to/irscode.pdf",
    "path/to/taxliensinvesting.pdf"
]
rag = GeneralizedRAG(model_name="AggregateModel", model_input="dolphin-llama3", pdf_source_files=pdf_paths)
question = "What are the tax implications of business expenses?"
answer = rag.ask_question(question=question)
print(f"Question: {question}")
print(f"Answer: {answer}")
'''


# Example usage with a single PDF
pdf_paths = [r"C:\Users\grc\Downloads\books_for_Train\Tax_code\usc26@118-64.pdf"]
rag_single = GeneralizedRAG(model_name="SingleModel", model_input="dolphin-llama3", pdf_source_files=pdf_paths)
question_single = "What are the tax implications of business expenses?"
answer_single = rag_single.ask_question(question=question_single)
print(f"Question: {question_single}")
print(f"Answer: {answer_single}")
