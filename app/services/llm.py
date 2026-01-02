from groq import Groq
from typing import List, Dict

class LLMService:
    def __init__(self, api_key: str):
        print("Initializing Groq LLM client")
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"
        print(f"LLM ready. Model: {self.model}")
    
    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """
        Generate answer using LLM with optional RAG context.
        
        Always uses knowledge base context when available, including procedural/escalation steps.
        """
        
        if not context_docs or len(context_docs) == 0:
            # NO CONTEXT - Pure LLM mode
            prompt = f"""You are a helpful assistant. Answer the following question based on your knowledge.

    Question: {query}

    Provide a clear and concise answer."""
        
        else:
            # WITH CONTEXT - RAG mode (ALWAYS use context)
            context = self._format_context(context_docs)
            
            prompt = f"""You are a helpful assistant for a company's internal knowledge base system.

    The information below is from the company's verified knowledge base and represents official procedures, processes, or solutions.

    IMPORTANT: Always use and follow the provided context in your answer. This may include technical steps, escalation procedures, contact information, meetings, or emails. All of these are valid and important company processes.

    Knowledge Base Context:
    {context}

    User Question: {query}

    Instructions:
    1. Base your answer on the provided knowledge base context
    2. Include ALL steps mentioned (technical, procedural, contacts, escalations)
    3. Present information clearly and in a user-friendly format
    4. Do not dismiss or skip any documented steps
    5. If the context provides a process to follow, explain it step-by-step

    Provide a clear and actionable answer based on the company's documented process:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def _format_context(self, docs: List[Dict]) -> str:
        context_parts = []
        for idx, doc in enumerate(docs, 1):
            similarity = (1 - doc.get('distance', 0)) * 100
            content = doc.get('document', '')
            context_parts.append(f"[Source {idx}] (Relevance: {similarity:.0f}%)\n{content}")
        
        return "\n\n".join(context_parts)
