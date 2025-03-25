# Microsoft GraphRAG Search API

![Python](https://img.shields.io/badge/Python-3776ab?logo=python&logoColor=fff)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=fff)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=fff)
![Ollama](https://img.shields.io/badge/Ollama-222?logo=ollama&logoColor=fff)

Microsoft GraphRAG Web API is a FastAPI-powered search API that offers multiple search functionalities (local, global, drift, and basic) via both GET and POST methods.

## Features

**Multiple Search Modes:**

Supports local, global, drift, and basic search endpoints.

**GET/POST Support:**

Each endpoint accepts both GET and POST requests.

**Domain-specific Configuration:**

Each domain loads its own configuration and data (Parquet files) from a specified directory.

**CORS Enabled:**

Configured CORS middleware allows local development access.

## Install

```sh
pip install .
```

## Configuration

The API settings can be overridden using an `environ.toml` file located in the current directory. The default configuration is:

```toml
GRAPHRAG_API_PORT = 3080
GRAPHRAG_ROOT_DIR = "."
```

Each domain must also include its own configuration file.

## Running the api

To start the API, simply run:

```sh
graphrag-api
```

The API will listen on host `0.0.0.0` and the port specified in `environ.toml` (default is 3080).

Alternatively, you can start it from your Python code:

```python
from graphrag_api import run
run()
```

## API Endpoints

All endpoints are prefixed with `/v1/search/{domain}`, where `{domain}` represents the directory name for domain-specific settings and data.

### Health Check

A simple endpoint to verify that the API is running.

#### GET /

```sh
curl http://localhost:3080/
```

**Response Body:**

```output
GraphRAG API is up and running
```

### Local Search

#### GET /v1/search/{domain}/local

Performs a search within a local context.

**Query Parameters:**

- `query` (required, string): The search query.
- `community_level` (optional, integer, default: 2): Level of community granularity in the Leiden hierarchy (higher values indicate smaller communities).
- `response_type` (optional, string, default: "Single paragraph"): Format of the response.

**Example:**

```sh
curl "http://localhost:3080/v1/search/xmascarol/local?query=Who%20is%20Scrooge%3F&community_level=2&response_type=Single%20paragraph"
```

#### POST /v1/search/{domain}/local

Same as the GET endpoint but accepts a JSON payload.

**Request Body Example:**

```json
{
  "query": "Who is Scrooge?",
  "community_level": 2,
  "response_type": "Single paragraph"
}
```

**Example:**

```sh
curl -X POST "http://localhost:3080/v1/search/xmascarol/local" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "Who is Scrooge?",
           "community_level": 2,
           "response_type": "Single paragraph"
         }'
```

**Response Body (GET/POST):**

```json
{"result":"Scrooge is a central character in Charles Dickens's novella \"A Christmas Carol.\".","query":"Who is Scrooge?"}
```

### Global Search

#### GET /v1/search/{domain}/global

Performs a search within a global context.

**Query Parameters:**

- `query` (required, string): The search query.
- `community_level` (optional, integer, default: 2): Level of community granularity in the Leiden hierarchy (higher values indicate smaller communities).
- `response_type` (optional, string, default: "Single paragraph"): Format of the response.

**Example:**

```sh
curl "http://localhost:3080/v1/search/xmascarol/global?query=What%20theme%20of%20Xmas%20Carol%3F&community_level=2&response_type=Single%20paragraph"
```

#### POST /v1/search/{domain}/global

Same as the GET endpoint but accepts a JSON payload.

**Request Body Example:**

```json
{
  "query": "What theme of Xmas Carol?",
  "community_level": 2,
  "response_type": "Single paragraph"
}
```

**Example:**

```sh
curl -X POST "http://localhost:3080/v1/search/xmascarol/global" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "What theme of Xmas Carol?",
           "community_level": 2,
           "response_type": "Single paragraph"
         }'
```

**Response Body (GET/POST):**

```json
{"result":"Scrooge's prominent role within society is evident, with connections to various institutions like Union Workhouses, Treadmill, Poor Law, and the Gentlemen.","query":"A theme of Xmas Carol?"}
```

### DRIFT Search

#### GET /v1/search/{domain}/drift

Performs a DRIFT context search, utilizing a full-content embedding store for richer content retrieval.

**Query Parameters:**

- `query` (required, string): The search query.
- `community_level` (optional, integer, default: 2): Level of community granularity in the Leiden hierarchy (higher values indicate smaller communities).
- `response_type` (optional, string, default: "Single paragraph"): Format of the response.

**Example:**

```sh
curl "http://localhost:3080/v1/search/xmascarol/drift?query=Who%20are%20the%20main%20characters%20in%20A%20Christmas%20Carol%3F&community_level=2&response_type=Single%20paragraph"
```

#### POST /v1/search/{domain}/drift

Same as the GET endpoint but accepts a JSON payload.

**Request Body Example:**

```json
{
  "query": "Who are the main characters in A Christmas Carol?",
  "community_level": 2,
  "response_type": "Single paragraph"
}
```

**Example:**

```sh
curl -X POST "http://localhost:3080/v1/search/xmascarol/drift" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "Who are the main characters in A Christmas Carol?",
           "community_level": 2,
           "response_type": "Single paragraph"
         }'
```

**Response Body (GET/POST):**

```json
{"result":"The main characters in *A Christmas Carol* are Ebenezer Scrooge, Bob Cratchit, Tiny Tim, and Jacob Marley.","query":"Who are the main characters in A Christmas Carol"}
```

### Basic Search

#### GET /v1/search/{domain}/basic

Provides the most straightforward search interface.

**Query Parameters:**

- `query` (required, string): The search query.
- `community_level` (optional, integer, default: 2): Level of community granularity in the Leiden hierarchy (higher values indicate smaller communities).

**Example:**

```sh
curl "http://localhost:3080/v1/search/xmascarol/basic?query=Who%20wrote%20A%20Xmas%20Carol%3F&community_level=2"
```

#### POST /v1/search/{domain}/basic

Same as the GET endpoint but accepts a JSON payload.

**Request Body Example:**

```json
{
  "query": "Who wrote A Xmas Carol?",
  "community_level": 2
}
```

**Example:**

```sh
curl -X POST "http://localhost:3080/v1/search/xmascarol/basic" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "Who wrote A Xmas Carol?",
           "community_level": 2
         }'
```

**Response Body:**

```json
{"result":"A Christmas Carol was written by Charles Dickens.","query":"Who wrote A Xmas Carol?"}
```

## License

[MIT License](LICENSE)

## Contributing

Don’t expect overly polite fluff – if you have improvements or fixes, open an issue or submit a pull request.
