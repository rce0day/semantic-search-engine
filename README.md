# Semantic Product Search

A vector-based semantic search engine for product data using sentence transformers and Qdrant vector database.

## Overview

This project implements a semantic search system for product data, allowing users to search for products using natural language queries. The system processes product descriptions, indexes them into vector embeddings, and provides a web interface for searching.

The example data in dpdata.jsonl is products scraped from amazon.

## Features

- **Semantic Search**: Uses sentence transformers to convert text into embeddings for semantic search capabilities
- **Vector Database**: Powered by Qdrant for efficient vector similarity search
- **Web Interface**: Clean, responsive UI for searching products
- **Admin Dashboard**: Monitor system health and view collection statistics
- **API Endpoints**: RESTful API for search and administration

## Tech Stack

- **Backend**: Python with FastAPI
- **Vector Database**: Qdrant
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Frontend**: HTML, JavaScript, Tailwind CSS
- **Deployment**: Docker and Docker Compose

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.8+ (for development without Docker)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/semantic-product-search.git
   cd semantic-product-search
   ```

2. Start the Qdrant vector database with Docker Compose:
   ```
   docker-compose up -d
   ```

3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

### Data Indexing

Example data provided in dpdata.jsonl, indexer uses this by default.

Run the indexer to create vector embeddings and store them in Qdrant:
```
python -m src.indexer
```

### Running the Search API

Start the FastAPI application:
```
python -m src.search_api
```

The search interface will be available at `http://localhost:8000`.

## API Endpoints

- `GET /search?query={query}&limit={limit}&offset={offset}` - Search for products
- `GET /health` - Check system health
- `GET /admin` - Get system statistics
- `GET /collection/{collection_name}/sample` - Get sample data from collection

## Project Structure

- `src/indexer.py` - Creates vector embeddings and indexes them into Qdrant
- `src/search_api.py` - FastAPI application with search and admin endpoints
- `src/static/` - Frontend HTML files
  - `index.html` - Search interface
  - `admin.html` - Admin dashboard
- `requirements.txt` - Python dependencies
- `docker-compose.yml` - Docker Compose configuration for Qdrant

## How It Works

1. **Indexing**:
   - Product data is read from the `dpdata.jsonl` file
   - Text from each product (title, description, features) is extracted and converted to vector embeddings
   - Embeddings and product data are stored in Qdrant database

2. **Searching**:
   - User enters a natural language query
   - Query is converted to a vector embedding
   - Qdrant performs vector similarity search to find the most relevant products
   - Results are displayed in the web interface with relevance scores

## Development

To run the application in development mode with hot-reloading:
```
uvicorn src.search_api:app --reload
```
