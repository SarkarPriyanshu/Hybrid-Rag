# Hybrid-RAG

Hybrid-RAG is a project that combines PostgreSQL and Pinecone VectorDB to extract relevant context, which is then used by a Large Language Model (LLM) to answer user queries intelligently.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Architecture](#architecture)  
3. [Installation](#installation)  
4. [Configuration](#configuration)  

## Project Overview

### Problem Statement
Many applications require context-aware responses to user queries, but traditional databases or keyword-based search methods often fail to provide relevant information when queries are phrased in natural language. Users may ask questions in varied ways, and retrieving the correct information quickly and accurately becomes a challenge.

### Solution
Hybrid-RAG (Retrieval-Augmented Generation) addresses this problem by combining:

- **PostgreSQL** for structured storage of domain-specific data.  
- **Pinecone VectorDB** for semantic search and retrieval of relevant context using vector embeddings.  
- **Large Language Model (LLM)** to generate intelligent and contextually accurate responses based on the retrieved information.

This hybrid approach ensures that user queries—whether exact keywords or full sentences—can be answered effectively, leveraging both structured data and semantic understanding.

## Architecture

Hybrid-RAG is structured to handle the end-to-end flow of data from extraction to user query response. Below is an overview of the architecture and how each component interacts:

### High-Level Flow

1. **Data Extraction & Preprocessing**
   - `app/utils/extract_data_from_source.py` fetches raw data from external sources (e.g., DailyMed).  
   - `app/utils/data_processess.py` cleans and preprocesses the raw text for insertion into the database.

2. **Database Storage**
   - `app/utils/setup_db.py` handles connections to **PostgreSQL**, storing structured data for efficient querying.  
   - Preprocessed data is inserted into PostgreSQL, enabling reliable retrieval.

3. **Vector Embeddings & Semantic Search**
   - `app/utils/vector_db.py` sets up connection to **Pinecone VectorDB** for semantic indexing.  
   - `app/models/embedding_*.py` contains multiple embedding models (BGE, MiniLM, MPNet) used to convert text into vectors for similarity search.

4. **Query Handling & LLM Response**
   - `main.py` serves as the entry point for user queries.  
   - `app/models/language_model.py` interacts with the LLM to generate answers using the context retrieved from PostgreSQL and Pinecone.

5. **Utilities**
   - `app/utils/custom_logging.py` provides logging across the application.  
   - `app/utils/string_to_dict_parser.py` includes helper functions for converting string data into dictionary structures.  

---

### File Structure

hybrid-rag/
├── config.py                  # Central configuration
├── docker-compose.yaml        # Docker Compose setup
├── Dockerfile                 # Docker image definition
├── main.py                    # Entry point for your app
├── requirements.txt           # Dependencies
├── setup.py                   # Package installer
└── app/
    ├── utils/
    │   ├── custom_logging.py
    │   ├── data_processess.py
    │   ├── extract_data_from_source.py
    │   ├── setup_db.py
    │   ├── string_to_dict_parser.py
    │   └── vector_db.py
    └── models/
        ├── embedding_bge_m3.py
        ├── embedding_miniLM_L6_v2.py
        ├── embedding_mpNet-base-v2.py
        └── language_model.py

Below is the architecture of Hybrid-RAG, showing how the components interact:

![Hybrid-RAG Architecture](https://github.com/SarkarPriyanshu/Hybrid-Rag/blob/main/architecture.png?raw=true)

## Installation

### 1. Create a `.env` file

Create a `.env` file in the root of your project (`hybrid-rag/.env`) to store sensitive configuration variables:

```env
# PostgreSQL Configuration
DB_USER=postgres
DB_PASSWORD=StrongPassw0rd!
DB_NAME=hybrid_rag
DB_URL=postgresql://postgres:StrongPassw0rd!@db:5432/hybrid_rag

# LLM Configuration
LLM_API_KEY=your_llm_api_key_here
LLM_MODEL_NAME=gemini-2.5-flash

# Vector DB Configuration
VECTOR_DB_API_KEY=your_vector_db_api_key_here
```

### 2. Build the Docker containers
From the project root (hybrid-rag/), run:
docker-compose build


### 3. Start the services
Run the services in detached mode:
docker-compose up -d

The rag-app service will wait for Postgres to be healthy before starting (via depends_on and healthcheck).

### 4. Check the running containers
docker ps

You should see something like:
CONTAINER ID   NAMES          IMAGE           PORTS
xxxxxxx        rag_app        hybrid-rag     0.0.0.0:8000->8000/tcp
xxxxxxx        postgres_db    postgres:16    5432/tcp


### 5. Access the API
Once running, your FastAPI app will be available at:
http://localhost:8000


## Configuration

Hybrid-RAG uses environment variables and a configuration file (`config.py`) to manage project settings. Proper configuration is required before starting the application.

### 1. Environment Variables

All sensitive credentials and configurable parameters are stored in a `.env` file in the project root. The application reads these variables when starting via Docker Compose.

Example `.env` file:

```env
# PostgreSQL Configuration
DB_USER=postgres
DB_PASSWORD=StrongPassw0rd!
DB_NAME=hybrid_rag
DB_URL=postgresql://postgres:StrongPassw0rd!@db:5432/hybrid_rag

# LLM Configuration
LLM_API_KEY=your_llm_api_key_here
LLM_MODEL_NAME=gemini-2.5-flash

# Vector DB Configuration
VECTOR_DB_API_KEY=your_vector_db_api_key_here
```

