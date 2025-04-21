from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from config.settings import settings

from langchain_ibm import WatsonxLLM
from config.settings import settings
from typing import Dict, List

class RelevanceChecker:
    def __init__(self):
        self.llm = WatsonxLLM(
            model_id=settings.IBM_MODEL_RELEVANCE,
            url=settings.WATSONX_URL,
            apikey=settings.WATSONX_API_KEY,
            project_id=settings.WATSONX_PROJECT_ID,
            params={
                "temperature": 0.1,
                "max_new_tokens": 1024,  # Increased for better context handling
                "min_new_tokens": 1,
                "top_p": 0.9,
                "repetition_penalty": 1.05,
                "stop_sequences": ["Human:", "Observation"],
                "truncate_input_tokens": 4096  # Added to handle long inputs
            }
        )

        self.prompt = ChatPromptTemplate.from_template(
            """
            You are given a user question and some passages from uploaded documents.
            
            Classify how well these passages address the user's question. 
            Choose exactly one of the following responses (respond ONLY with that label):
            
            1) "CAN_ANSWER": The passages contain enough explicit info to fully answer the question.
            2) "PARTIAL": The passages mention or discuss the question's topic (e.g., relevant years, facility names)
            but do not provide all the data or details needed for a complete answer.
            3) "NO_MATCH": The passages do not discuss or mention the question's topic at all.
            
            Important: If the passages mention or reference the topic or timeframe of the question in ANY way,
            even if incomplete, you should respond "PARTIAL", not "NO_MATCH".
            
            Question: {question}
            Passages: {document_content}
            
            Respond ONLY with "CAN_ANSWER", "PARTIAL", or "NO_MATCH".
            """
        )

        self.chain = self.prompt | self.llm | StrOutputParser()

    def check(self, question: str, retriever, k=3) -> str:
        """
        1. Retrieve the top-k document chunks from the global retriever.
        2. Combine them into a single text string.
        3. Pass that text + question to the LLM chain for classification.
        
        Returns: "CAN_ANSWER" or "PARTIAL" or "NO_MATCH".
        """

        print(f"[DEBUG] RelevanceChecker.check called with question='{question}' and k={k}")
        
        # Retrieve doc chunks from the ensemble retriever
        top_docs = retriever.invoke(question)
        if not top_docs:
            print("[DEBUG] No documents returned from retriever.invoke(). Classifying as NO_MATCH.")
            return "NO_MATCH"

        # Print how many docs were retrieved in total
        print(f"[DEBUG] Retriever returned {len(top_docs)} docs. Now taking top {k} to feed LLM.")

        # Show a quick snippet of each chunk for debugging
        for i, doc in enumerate(top_docs[:k]):
            snippet = doc.page_content[:200].replace("\n", "\\n")
            print(f"[DEBUG] Chunk #{i+1} preview (first 200 chars): {snippet}...")

        # Combine the top k chunk texts into one string
        document_content = "\n\n".join(doc.page_content for doc in top_docs[:k])
        print(f"[DEBUG] Combined text length for top {k} chunks: {len(document_content)} chars.")

        # Call the LLM
        response = self.chain.invoke({
            "question": question, 
            "document_content": document_content
        }).strip()
        
        print(f"[DEBUG] LLM raw classification response: '{response}'")


        # Extract classification from response
        classification = self._extract_classification_label(response)
        print(f"[DEBUG] Extracted classification: '{classification}'.")

        return classification

        # # Convert to uppercase, check if it's one of our valid labels
        # classification = response.upper()
        # valid_labels = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}
        # if classification not in valid_labels:
        #     print("[DEBUG] LLM did not respond with a valid label. Forcing 'NO_MATCH'.")
        #     classification = "NO_MATCH"
        # else:
        #     print(f"[DEBUG] Classification recognized as '{classification}'.")

        # return classification
    
    def _extract_classification_label(self, response: str) -> str:
        """
        Extract a valid classification label from the LLM response.
        
        Looks for "CAN_ANSWER", "PARTIAL", or "NO_MATCH" within the response text.
        If multiple labels are found, returns the first one.
        If no valid label is found, returns "NO_MATCH" as fallback.
        """
        valid_labels = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}
        
        # First check: Does the full response (uppercase) match a valid label?
        full_response_upper = response.upper()
        if full_response_upper in valid_labels:
            return full_response_upper
        
        # Extra robust: Check for a valid label at the start of any line
        import re
        for line in response.splitlines():
            match = re.match(r"^(CAN_ANSWER|PARTIAL|NO_MATCH)$", line.strip(), re.IGNORECASE)
            if match:
                candidate = match.group(1).upper()
                if candidate in valid_labels:
                    return candidate
        
        # Second check: Look for "## Response: LABEL" or similar patterns
        label_patterns = [
            r"## Response: (\w+)",              # Matches "## Response: CAN_ANSWER"
            r"Response: (\w+)",                 # Matches "Response: CAN_ANSWER"
            r"classification: (\w+)",           # Matches "classification: CAN_ANSWER"
            r"classify as (\w+)",               # Matches "classify as CAN_ANSWER"
        ]
        
        for pattern in label_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                candidate = matches[0].upper()
                if candidate in valid_labels:
                    return candidate
        
        # Third check: Simply look for any of the valid labels in the response
        for label in valid_labels:
            if label in full_response_upper:
                return label
        
        # If no valid label is found, default to "NO_MATCH"
        print("[DEBUG] No valid classification label found in LLM response. Defaulting to 'NO_MATCH'.")
        return "NO_MATCH"
