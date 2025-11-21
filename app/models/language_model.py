# app/models/language_model.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from config import config
from app.utils.custom_logging import logger

# ----------------------
# Initialize LLM
# ----------------------
llm_model_name = config.components["llm"].model_name
api_key = config.components["llm"].api_key
llm = ChatGoogleGenerativeAI(model=llm_model_name, temperature=0, api_key=api_key)

# ----------------------
# Build final LLM prompt
# ----------------------
def build_prompt(db_answer, vector_context, user_query):
    """
    Combine DB answer, vector context, and the original user query
    into a single prompt for LLM generation.
    """
    prompt = f"""
        You are an expert helpful assistant.

        User query:
        {user_query}

        PostgreSQL database context:
        {db_answer}

        Vector search context:
        {vector_context}

        Using the above information, provide a concise, informative, and accurate answer to the user's query.
    """
    return prompt.strip()


def query_google_llm(db_answer, vector_context, user_query):
    """
    Send the prompt to Google GenAI LLM and return the generated answer.
    """
    prompt = build_prompt(db_answer, vector_context, user_query)
    logger.info(f"[LLM PROMPT] Prompt length: {len(prompt)} chars")
    logger.info(f"[LLM PROMPT] Prompt: {prompt}")

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        logger.error(f"[LLM ERROR] Failed to generate answer: {str(e)}")
        return f"Error generating answer: {str(e)}"


# ----------------------
# Extract main entity/keyword
# ----------------------
def extract_main_entity(user_query):
    """
    Given a natural language query, return the main entity/keyword
    that should be used to query the database.
    """
    prompt = f"""
        Extract the main entity or keyword from the user's query.
        Only return the main word or phrase that represents the item to search for.
        User query: "{user_query}"
        Return a single word or short phrase, nothing else.
    """
    logger.info(f"[LLM ENTITY] Extracting entity from query: {user_query}")

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        entity = response.content.strip()
        logger.info(f"[LLM ENTITY] Extracted entity: {entity}")
        return entity
    except Exception as e:
        logger.error(f"[LLM ENTITY ERROR] Failed to extract entity: {str(e)}")
        return None
