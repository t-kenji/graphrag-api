"""Microsoft GraphRAG web API package.

This module implements the Microsoft GraphRAG query search web API using FastAPI.
It provides endpoints for local, global, drift, and basic search functionalities.
"""

import json
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import pandas as pd
import tomllib
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from graphrag.config.load_config import load_config
from graphrag.config.resolve_path import resolve_paths
from graphrag.logger.print_progress import PrintProgressLogger
from graphrag.query.factory import (
    get_basic_search_engine,
    get_drift_search_engine,
    get_global_search_engine,
    get_local_search_engine,
)
from graphrag.query.indexer_adapters import (
    read_indexer_communities,
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_report_embeddings,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.vector_stores.lancedb import LanceDBVectorStore

__version__ = version(__package__)

# Initialize a logger instance (prints progress information)
logger = PrintProgressLogger("")


class DictDotNotation(dict):
    """Dictionary subclass that supports attribute-style access.

    This class allows accessing dictionary keys as attributes.
    For example:
        d = DictDotNotation({"a": 1})
        print(d.a)  # prints 1
    """

    def __init__(self, *args: list | None, **kwargs: dict | None) -> None:
        """Initialize the dictionary with optional positional and keyword arguments."""
        super().__init__(*args, **kwargs)
        self.__dict__ = self


# Default configuration parameters.
config = DictDotNotation({
    "GRAPHRAG_API_PORT": 3080,
    "GRAPHRAG_API_KEY": "<YOUR API KEY>",
    "GRAPHRAG_LLM_API_BASE": "https://api.oepnai.com/v1",
    "GRAPHRAG_LLM_MODEL": "gpt-4o",
    "GRAPHRAG_EMBEDDING_API_BASE": "https://api.openai.com/v1",
    "GRAPHRAG_EMBEDDING_MODEL": "text-embedding-3-large",
    "GRAPHRAG_ROOT_DIR": ".",
})

# Override default config values if environ.toml exists.
config_file = Path("environ.toml")
if config_file.exists():
    with config_file.open("rb") as f:
        for k, v in tomllib.load(f).items():
            if k in config:
                config[k] = v

# Create FastAPI app instance.
app = FastAPI()

# Add CORS middleware to allow requests from the specified origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"http://localhost:{config.GRAPHRAG_API_PORT}"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_parquet(path: Path) -> pd.DataFrame | None:
    """Read a Parquet file and return its DataFrame.

    Args:
        path (Path): The path to the Parquet file.

    Returns:
        pd.DataFrame | None: DataFrame if the file exists, otherwise None.
    """
    return pd.read_parquet(path) if path.is_file() else None


def _tostring(x: any) -> str:
    """Convert response data to a string representation.

    If the input is a dictionary or list, it will be converted to a JSON string.
    Otherwise, the standard string representation is returned.

    Args:
        x (any): The input data to convert.

    Returns:
        str: The string representation of the input.
    """
    return json.dumps(x) if isinstance(x, dict | list) else str(x)


@app.get("/v1/search/{domain}/local")
async def local_search_v1(
    domain: str,
    query: Annotated[str, Query(description="Search query for local context")] = ...,
    community_level: Annotated[int, Query(
        description=("The community level in the Leiden community hierarchy from "
                     "which to load community reports. Higher values represent "
                     "reports from smaller communities.")
    )] = 2,
) -> Response:
    """Perform a local context search for a specified domain.

    This endpoint reads the necessary parquet files, constructs the local search
    engine using the provided domain configuration, and returns the search result.

    Args:
        domain (str): The domain (directory name) to search within.
        query (str): The search query string.
        community_level (int, optional): The community level for filtering reports.
            Defaults to 2.

    Returns:
        Response: A JSON response containing the search result and the original query.
    """
    try:
        # Resolve the domain root directory and load its configuration.
        domain_root = Path(f"{config.GRAPHRAG_ROOT_DIR}/{domain}").resolve()
        domain_config = load_config(domain_root)
        resolve_paths(domain_config)

        # Define the output directory based on the storage configuration.
        output_dir = Path(domain_config.storage.base_dir)

        # Read required parquet files.
        final_nodes = _read_parquet(output_dir / "create_final_nodes.parquet")
        final_entities = _read_parquet(output_dir / "create_final_entities.parquet")
        final_community_reports = _read_parquet(output_dir / "create_final_community_reports.parquet")
        final_relationships = _read_parquet(output_dir / "create_final_relationships.parquet")
        final_covariates = _read_parquet(output_dir / "create_final_covariates.parquet")
        final_text_units = _read_parquet(output_dir / "create_final_text_units.parquet")

        # Build indexers for entities, relationships, claims, reports, and text units.
        entities = read_indexer_entities(final_nodes, final_entities, community_level)
        relationships = read_indexer_relationships(final_relationships)
        claims = read_indexer_covariates(final_covariates) if final_covariates is not None else []
        community_reports = read_indexer_reports(final_community_reports, final_nodes, community_level)
        text_units = read_indexer_text_units(final_text_units)

        # Connect to the description embedding store using LanceDB.
        description_embedding_store = LanceDBVectorStore(collection_name="default-entity-description")
        description_embedding_store.connect(db_uri=output_dir / "lancedb")

        # Initialize the local search engine.
        search_engine = get_local_search_engine(
            config=domain_config,
            reports=community_reports,
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            covariates={"claims": claims},
            response_type="Single paragraph",
            description_embedding_store=description_embedding_store,
        )

        # Execute the asynchronous search.
        result = await search_engine.asearch(query)

        # Return the search result as a JSON response.
        return JSONResponse(
            content={
                "result": _tostring(result.response),
                "query": query,
            }
        )
    except Exception as e:
        # In case of error, return an HTTP 500 response with the error detail.
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/v1/search/{domain}/global")
async def global_search_v1(
    domain: str,
    query: Annotated[str, Query(description="Search query for global context")] = ...,
    community_level: Annotated[int, Query(
        description=("The community level in the Leiden community hierarchy from "
                     "which to load community reports. Higher values represent "
                     "reports from smaller communities.")
    )] = 2,
) -> Response:
    """Perform a global context search for a specified domain.

    This endpoint loads the domain configuration and parquet data required for a global
    search, constructs the global search engine, and returns the search result.

    Args:
        domain (str): The domain (directory name) to search within.
        query (str): The search query string.
        community_level (int, optional): The community level for filtering reports.
            Defaults to 2.

    Returns:
        Response: A JSON response containing the search result and the original query.
    """
    try:
        # Resolve the domain root directory and load its configuration.
        domain_root = Path(f"{config.GRAPHRAG_ROOT_DIR}/{domain}").resolve()
        domain_config = load_config(domain_root)
        resolve_paths(domain_config)

        # Define the output directory based on the storage configuration.
        output_dir = Path(domain_config.storage.base_dir)

        # Read required parquet files.
        final_nodes = _read_parquet(output_dir / "create_final_nodes.parquet")
        final_entities = _read_parquet(output_dir / "create_final_entities.parquet")
        final_community_reports = _read_parquet(output_dir / "create_final_community_reports.parquet")
        final_communities = _read_parquet(output_dir / "create_final_communities.parquet")

        # Build indexers for entities, reports, and communities.
        entities = read_indexer_entities(final_nodes, final_entities, community_level)
        community_reports = read_indexer_reports(final_community_reports, final_nodes, community_level)
        communities = read_indexer_communities(final_communities, final_nodes, final_community_reports)

        # Initialize the global search engine.
        search_engine = get_global_search_engine(
            config=domain_config,
            reports=community_reports,
            entities=entities,
            communities=communities,
            response_type="Single paragraph",
        )

        # Execute the asynchronous search.
        result = await search_engine.asearch(query)

        # Return the search result as a JSON response.
        return JSONResponse(content={"result": _tostring(result.response), "query": query})
    except Exception as e:
        # In case of error, return an HTTP 500 response with the error detail.
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/v1/search/{domain}/drift")
async def drift_search_v1(
    domain: str,
    query: Annotated[str, Query(description="Search query for drift context")] = ...,
    community_level: Annotated[int, Query(
        description=("The community level in the Leiden community hierarchy from "
                     "which to load community reports. Higher values represent "
                     "reports from smaller communities.")
    )] = 2,
) -> Response:
    """Perform a drift context search for a specified domain.

    This endpoint loads the necessary parquet data and connects to both description and
    full-content embedding stores. It constructs the drift search engine and returns the search result.

    Args:
        domain (str): The domain (directory name) to search within.
        query (str): The search query string.
        community_level (int, optional): The community level for filtering reports.
            Defaults to 2.

    Returns:
        Response: A JSON response containing the search result and the original query.
    """
    try:
        # Resolve the domain root directory and load its configuration.
        domain_root = Path(f"{config.GRAPHRAG_ROOT_DIR}/{domain}").resolve()
        domain_config = load_config(domain_root)
        resolve_paths(domain_config)

        # Define the output directory based on the storage configuration.
        output_dir = Path(domain_config.storage.base_dir)

        # Read required parquet files.
        final_nodes = _read_parquet(output_dir / "create_final_nodes.parquet")
        final_entities = _read_parquet(output_dir / "create_final_entities.parquet")
        final_relationships = _read_parquet(output_dir / "create_final_relationships.parquet")
        final_community_reports = _read_parquet(output_dir / "create_final_community_reports.parquet")
        final_text_units = _read_parquet(output_dir / "create_final_text_units.parquet")

        # Build indexers for entities, relationships, reports, and text units.
        entities = read_indexer_entities(final_nodes, final_entities, community_level)
        relationships = read_indexer_relationships(final_relationships)
        community_reports = read_indexer_reports(final_community_reports, final_nodes, community_level)
        text_units = read_indexer_text_units(final_text_units)

        # Connect to the description embedding store.
        description_embedding_store = LanceDBVectorStore(collection_name="default-entity-description")
        description_embedding_store.connect(db_uri=output_dir / "lancedb")

        # Connect to the full-content embedding store.
        full_content_embedding_store = LanceDBVectorStore(collection_name="default-community-full_content")
        full_content_embedding_store.connect(db_uri=output_dir / "lancedb")

        # Populate the full-content embedding store with report embeddings.
        read_indexer_report_embeddings(community_reports, full_content_embedding_store)

        # Initialize the drift search engine.
        search_engine = get_drift_search_engine(
            config=domain_config,
            reports=community_reports,
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            description_embedding_store=description_embedding_store,
            response_type="Single paragraph",
        )

        # Execute the asynchronous search.
        result = await search_engine.asearch(query=query)

        # Return the search result as a JSON response.
        return JSONResponse(content={"result": _tostring(result.response), "query": query})
    except Exception as e:
        # In case of error, return an HTTP 500 response with the error detail.
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/v1/search/{domain}/basic")
async def basic_search_v1(
    domain: str,
    query: Annotated[str, Query(description="Search query for basic context")] = ...,
    community_level: Annotated[int, Query(
        description=("The community level in the Leiden community hierarchy from "
                     "which to load community reports. Higher values represent "
                     "reports from smaller communities.")
    )] = 2,
) -> Response:
    """Perform a basic context search for a specified domain.

    This endpoint loads the necessary parquet data and connects to multiple embedding
    stores before constructing the basic search engine and returning the search result.

    Args:
        domain (str): The domain (directory name) to search within.
        query (str): The search query string.
        community_level (int, optional): The community level for filtering reports.
            Defaults to 2.

    Returns:
        Response: A JSON response containing the search result and the original query.
    """
    try:
        # Resolve the domain root directory and load its configuration.
        domain_root = Path(f"{config.GRAPHRAG_ROOT_DIR}/{domain}").resolve()
        domain_config = load_config(domain_root)
        resolve_paths(domain_config)

        # Define the output directory based on the storage configuration.
        output_dir = Path(domain_config.storage.base_dir)

        # Read required parquet files.
        final_nodes = _read_parquet(output_dir / "create_final_nodes.parquet")
        final_community_reports = _read_parquet(output_dir / "create_final_community_reports.parquet")
        final_text_units = _read_parquet(output_dir / "create_final_text_units.parquet")

        # Build indexers for community reports and text units.
        community_reports = read_indexer_reports(final_community_reports, final_nodes, community_level)
        text_units = read_indexer_text_units(final_text_units)

        # Connect to the description embedding store.
        description_embedding_store = LanceDBVectorStore(collection_name="default-entity-description")
        description_embedding_store.connect(db_uri=output_dir / "lancedb")

        # Connect to the full-content embedding store.
        full_content_embedding_store = LanceDBVectorStore(collection_name="default-community-full_content")
        full_content_embedding_store.connect(db_uri=output_dir / "lancedb")

        # Connect to the text unit text embedding store.
        text_unit_text_embedding_store = LanceDBVectorStore(collection_name="default-text_unit-text")
        text_unit_text_embedding_store.connect(db_uri=output_dir / "lancedb")

        # Populate the full-content embedding store with report embeddings.
        read_indexer_report_embeddings(community_reports, full_content_embedding_store)

        # Initialize the basic search engine.
        search_engine = get_basic_search_engine(
            config=domain_config,
            text_units=text_units,
            text_unit_embeddings=description_embedding_store,
        )

        # Execute the asynchronous search.
        result = await search_engine.asearch(query=query)

        # Return the search result as a JSON response.
        return JSONResponse(content={"result": _tostring(result.response), "query": query})
    except Exception as e:
        # In case of error, return an HTTP 500 response with the error detail.
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/")
async def status() -> str:
    """Return the status of the GraphRAG API.

    Returns:
        str: A simple status message.
    """
    return "GraphRAG API is up and running"


def run() -> None:
    """Run the GraphRAG API service using uvicorn.

    This function starts the FastAPI application on all available IPs
    with the port specified in the configuration.
    """
    uvicorn.run(app, host="0.0.0.0", port=config.GRAPHRAG_API_PORT)
