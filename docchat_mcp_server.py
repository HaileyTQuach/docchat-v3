from mcp.server.fastmcp import FastMCP
from document_processor.file_handler import DocumentProcessor
from agents.research_agent import ResearchAgent
from agents.workflow import AgentWorkflow
from dotenv import load_dotenv
import os
import sys
from loguru import logger
from retriever.builder import RetrieverBuilder

load_dotenv()

mcp = FastMCP("docchat")

processor = DocumentProcessor()
research_agent = ResearchAgent()
workflow = AgentWorkflow()

retriever_builder = RetrieverBuilder()

logger.add(sys.stderr, level="INFO")

@mcp.tool()
async def summarize_documents(file_paths: list[str]) -> dict:
    """
    Summarize each document separately. Returns a dict mapping file name to summary.
    """
    summaries = {}
    for path in file_paths:
        try:
            with open(path, "rb") as f:
                chunks = processor._process_file(f)
            # Use a summarization prompt
            summary_prompt = "Summarize the following document in a concise and informative way."
            result = research_agent.generate(summary_prompt, chunks)
            summaries[path] = result["draft_answer"]
        except Exception as e:
            summaries[path] = f"Error: {str(e)}"
    return summaries

@mcp.tool()
async def answer_question(file_paths: list[str], question: str) -> dict:
    """
    Given a list of document file paths and a question, return an answer and verification report.
    """
    # 1. Process documents
    files = [open(path, "rb") for path in file_paths]
    chunks = processor.process(files)
    for f in files:
        f.close()
    # 2. Build retriever
    retriever = retriever_builder.build_hybrid_retriever(chunks)
    # 3. Run workflow
    result = workflow.full_pipeline(question=question, retriever=retriever)
    return {
        "answer": result["draft_answer"],
        "verification_report": result["verification_report"]
    }

if __name__ == "__main__":
    mcp.run(transport="stdio") 