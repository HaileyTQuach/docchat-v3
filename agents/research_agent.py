from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, List
from langchain.schema import Document
from config.settings import settings
import logging

from langchain_ibm import WatsonxLLM
from typing import Dict, Any, List


logger = logging.getLogger(__name__)

class ResearchAgent:
    def __init__(self):
        """Initialize the research agent with the OpenAI model."""
        self.llm = WatsonxLLM(
            model_id=settings.IBM_MODEL_RESEARCH,
            url=settings.WATSONX_URL,
            apikey=settings.WATSONX_API_KEY,
            project_id=settings.WATSONX_PROJECT_ID,
            params={
                "temperature": 0.3,
                "max_new_tokens": 1024,  # Increased for more comprehensive answers
                "min_new_tokens": 1,
                "top_p": 0.7,
                "repetition_penalty": 1.05,
                "truncate_input_tokens": 8192 # Increased to handle longer context
            }
        )

        self.prompt = ChatPromptTemplate.from_template(
            """You are a research assistant that answers questions based on provided context.
            
            Answer the following question based on the provided context. Be precise and factual.
            
            DO NOT add "Human:" or any other chat markers at the end of your response.
            
            Question: {question}
            
            Context:
            {context}
            
            If the context is insufficient, respond with: "I cannot answer this question based on the provided documents."
            """
        )

        self.chain = self.prompt | self.llm | StrOutputParser()
        
    def generate(self, question: str, documents: List[Document]) -> Dict:
        """Generate an initial answer using the provided documents."""
        context = "\n\n".join([doc.page_content for doc in documents])
        
        chain = self.prompt | self.llm | StrOutputParser()
        try:
            answer = chain.invoke({
                "question": question,
                "context": context
            })
            
            # Clean up any unwanted tokens
            if answer.endswith("Human:"):
                answer = answer.replace("Human:", "").strip()
                
            logger.info(f"Generated answer: {answer}")
            logger.info(f"Context used: {context}")
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

        print(f"[DEBUG] Generated answer: {answer}")
        
        return {
            "draft_answer": answer,
            "context_used": context
        }