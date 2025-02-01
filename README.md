# Microsoft GraphRAG Search API

![Python](https://img.shields.io/badge/Python-3776ab?logo=python&logoColor=fff)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=fff)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=fff)
![Ollama](https://img.shields.io/badge/Ollama-222?logo=ollama&logoColor=fff)

This Python package allows you to search Microsoft GraphRAG via Web API.

## Install

```sh
pip install .
```

## Run the api

```sh
graphrag-api
```

Configure settings in the `environ.toml` file.

**`environ.toml` example:**

```toml
GRAPHRAG_API_PORT = 3080
GRAPHRAG_API_KEY = "<YOUR API KEY>"
GRAPHRAG_LLM_API_BASE = "https://api.oepnai.com/v1"
GRAPHRAG_LLM_MODEL = "gpt-4o"
GRAPHRAG_EMBEDDING_API_BASE = "https://api.openai.com/v1"
GRAPHRAG_EMBEDDING_MODEL = "text-embedding-3-large"
GRAPHRAG_ROOT_DIR = "."
```

## API Reference

The API to the GraphRAG Search API is described below.

### Local Search

**Request:**

`GET /v1/search/local/{domain}`

query:
> Who is Scrooge?

```sh
curl http://localhost:3080/v1/search/local/xmascarol?query=Who%20is%20Scrooge%3F
```

**Response Body:**

```json
{"result":"Scrooge is a central character in Charles Dickens's novella \"A Christmas Carol.\".","query":"Who is Scrooge?"}
```

### Global Search

**Request:**

`GET /v1/search/global/{domain}`

query:
> A theme of Xmas Carol?

```sh
curl http://localhost:3080/v1/search/global/xmascarol?query=What%20theme%20of%20Xmas%20Carol%3F
```

**Response Body:**

```json
{"result":"Scrooge's prominent role within society is evident, with connections to various institutions like Union Workhouses, Treadmill, Poor Law, and the Gentlemen.","query":"A theme of Xmas Carol?"}
```

### DRIFT Search

**Request:**

`GET /v1/search/drift/{domain}`

query:
> Who are the main characters in A Christmas Carol

```sh
curl http://localhost:3080/v1/search/drift/xmascarol?query=Who%20are%20the%20main%20characters%20in%20A%20Christmas%20Carol
```

**Response Body:**

```json
{"result":"The main characters in *A Christmas Carol* are Ebenezer Scrooge, Bob Cratchit, Tiny Tim, and Jacob Marley.","query":"Who are the main characters in A Christmas Carol"}
```

### Basic Search

**Request:**

`GET /v1/search/basic/{domain}`

query:
> Who wrote A Xmas Carol?

```sh
curl http://localhost:3080/v1/search/basic/xmascarol?query=Who%20wrote%20A%20Xmas%20Carol%3F
```

**Response Body:**

```json
{"result":"A Christmas Carol was written by Charles Dickens.","query":"Who wrote A Xmas Carol?"}
```
