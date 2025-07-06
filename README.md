# Retrieval-Augmented Q&A with MongoDB Atlas

This project demonstrates a simple Retrieval-Augmented Generation (RAG) workflow using MongoDB Atlas for vector search and OpenAI for question answering. It includes a document ingestion pipeline, a FastAPI service for querying the indexed documents and an example PDF to get started.

## Features

- **Document ingestion** – Cleans PDF pages, tags metadata with OpenAI, chunks the text and stores vector embeddings in MongoDB Atlas.
- **Question answering** – Queries the vector store using similarity search and generates answers with OpenAI's chat model.
- **FastAPI service** – Exposes a `/api/query` endpoint for asking questions and a `/api/stats` endpoint to see basic usage metrics.
- **Web Interface** – A [companion web UI](https://github.com/nmo-genio/mongodb-assistant-ui) is available for a more interactive experience.

## Requirements

- Python 3.13+
- A MongoDB Atlas cluster with a vector search index
- An OpenAI API key

Install the dependencies with:

```bash
pip install -e .
```

## Environment variables

Create a `.env` file based on `.env.example` and provide the following variables:

- `OPENAI_API_KEY` – your OpenAI API key
- `MONGODB_ATLAS_URI` – MongoDB Atlas connection string
- `MONGODB_DB_NAME` – database to store embeddings
- `MONGODB_COLLECTION_NAME` – collection to store embeddings
- `MONGODB_SEARCH_INDEX` – name of the vector search index (required by the query service)

## Ingesting documents

Run the ingestion pipeline to load a PDF and store its embeddings:

```bash
python src/rag_architecture/main.py path/to/document.pdf
```

By default it processes `sample_files/mongodb.pdf`.

## Running the API

Start the FastAPI application using Uvicorn:

```bash
uvicorn rag_architecture.app:app --reload
```

Send a POST request to `/api/query` with a JSON body containing a `question` field to receive an answer. The service also tracks how many questions were asked and how many answers were successfully generated.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
