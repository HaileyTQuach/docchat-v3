from langchain_core.output_parsers import StrOutputParser
from typing import Dict, List
from langchain.schema import Document
from config.settings import settings
import logging
from langchain_ibm import WatsonxLLM
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

logger = logging.getLogger(__name__)

class VerificationAgent:
    def __init__(self):
        self.llm = WatsonxLLM(
            model_id=settings.IBM_MODEL_VERIFICATION,
            url=settings.WATSONX_URL,
            apikey=settings.WATSONX_API_KEY,
            project_id=settings.WATSONX_PROJECT_ID,
            params={
                "temperature": 0.1,
                "max_new_tokens": 512,
                "min_new_tokens": 1,
                "top_p": 0.7,
                "repetition_penalty": 1.05,
                "stop_sequences": ["Human:", "Observation"]
            }
        )

        system_prompt = """Respond to the human as helpfully and accurately as possible. 
                Verify the following answer against the provided context. Check for:
                1. Direct factual support (YES/NO)
                2. Unsupported claims (list)
                3. Contradictions (list)
                4. Relevance to the question (YES/NO)
                
                Respond in this format:
                Supported: YES/NO
                Unsupported Claims: [items]
                Contradictions: [items]
                Relevant: YES/NO
                
                Answer: {answer}
                Context: {context}
                """
        human_prompt = """{answer}"""

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", human_prompt),
            ]
        )

        
    def check(self, answer: str, documents: List[Document]) -> Dict:
        """Verify the answer against the provided documents."""
        context = "\n\n".join([doc.page_content for doc in documents])
        
        chain = self.prompt | self.llm | StrOutputParser()
        try:
            verification = chain.invoke({
                "answer": answer,
                "context": context
            })
            logger.info(f"Verification report: {verification}")
            logger.info(f"Context used: {context}")
        except Exception as e:
            logger.error(f"Error verifying answer: {e}")
            raise
        
        return {
            "verification_report": verification,
            "context_used": context
        }