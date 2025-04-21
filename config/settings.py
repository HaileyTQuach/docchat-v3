from pydantic_settings import BaseSettings
from .constants import MAX_FILE_SIZE, MAX_TOTAL_SIZE, ALLOWED_TYPES
import os

class Settings(BaseSettings):
    # Required settings for LLM
    # Replace OpenAI with IBM watsonx.ai credentials
    # OPENAI_API_KEY: str
    WATSONX_API_KEY: str
    WATSONX_URL: str = "https://us-south.ml.cloud.ibm.com"
    WATSONX_PROJECT_ID: str
    
    # IBM model IDs for different use cases
    # Reference models available: https://ibm.github.io/watsonx-ai-python-sdk/model_definitions.html
    IBM_MODEL_MAIN: str = "ibm/granite-3-3-8b-instruct" # Default model if agent-specific models not set
    
    # Agent-specific models (can be overridden in .env)
    # IBM_MODEL_RESEARCH: str = "meta-llama/llama-3-3-70b-instruct"  # Model for the research agent
    IBM_MODEL_RESEARCH: str = "ibm/granite-3-2-8b-instruct"  # Model for the research agent
    IBM_MODEL_VERIFICATION: str = "ibm/granite-3-3-8b-instruct"  # Model for verification agent
    IBM_MODEL_RELEVANCE: str = "ibm/granite-3-3-8b-instruct"  # Model for relevance checker
    
    # Embedding model
    IBM_MODEL_EMBEDDINGS: str = "ibm/slate-125m-english-rtrvr" # For embedding vectors
    
    # Optional settings with defaults
    MAX_FILE_SIZE: int = MAX_FILE_SIZE
    MAX_TOTAL_SIZE: int = MAX_TOTAL_SIZE
    ALLOWED_TYPES: list = ALLOWED_TYPES

    # Database settings
    CHROMA_DB_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "documents"

    # Retrieval settings
    VECTOR_SEARCH_K: int = 10
    HYBRID_RETRIEVER_WEIGHTS: list = [0.4, 0.6]

    # Logging settings
    LOG_LEVEL: str = "INFO"

    # New cache settings with type annotations
    CACHE_DIR: str = "document_cache"
    CACHE_EXPIRE_DAYS: int = 7

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()