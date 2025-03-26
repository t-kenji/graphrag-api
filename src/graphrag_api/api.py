"""Microsoft GraphRAG web API package.

This module implements the Microsoft GraphRAG query search web API using FastAPI.
It provides endpoints for local, global, drift, and basic search functionalities.
Both GET and POST methods are supported.
"""

import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import tomllib
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from graphrag.config.load_config import GraphRagConfig, load_config
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
from pydantic import BaseModel

# Initialize logger instance.
logger = PrintProgressLogger("")

# -----------------------------------------------------------------------------
# Pydantic models for request/response
# -----------------------------------------------------------------------------

class SearchRequest(BaseModel):
    """Request model for search endpoints."""
    query: str
    community_level: int = 2
    response_type: str = "Single paragraph"


class SearchResponse(BaseModel):
    """Response model for search endpoints."""
    result: str
    query: str

# -----------------------------------------------------------------------------
# Utility: Dictionary with attribute-style access.
# -----------------------------------------------------------------------------

class DictDotNotation(dict):
    """Dictionary subclass that supports attribute-style access.

    E.g., d = DictDotNotation({"a": 1}); print(d.a) -> 1.
    """

    def __init__(self, *args: list, **kwargs: dict) -> None:
        """Initializer."""
        super().__init__(*args, **kwargs)
        self.__dict__ = self

# -----------------------------------------------------------------------------
# Default configuration
# -----------------------------------------------------------------------------

app_config = DictDotNotation({
    "GRAPHRAG_API_PORT": 3080,
    "GRAPHRAG_ROOT_DIR": ".",
})

# Override default config if environ.toml exists.
config_file = Path("environ.toml")
if config_file.exists():
    with config_file.open("rb") as f:
        for k, v in tomllib.load(f).items():
            if k in app_config:
                app_config[k] = v

# -----------------------------------------------------------------------------
# FastAPI app initialization and CORS setup
# -----------------------------------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"http://localhost:{app_config.GRAPHRAG_API_PORT}"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Helper functions for file loading and store initialization
# -----------------------------------------------------------------------------

def _read_parquet(path: Path) -> pd.DataFrame | None:
    """Read a Parquet file and return its DataFrame.

    Args:
        path (Path): The path to the Parquet file.

    Returns:
        pd.DataFrame | None: DataFrame if file exists, else None.
    """
    return pd.read_parquet(path) if path.is_file() else None


def _tostring(x: any) -> str:
    """Convert input data to a string representation.

    If input is a dict or list, return its JSON string; otherwise, use str().

    Args:
        x (Any): Input data.

    Returns:
        str: String representation.
    """
    return json.dumps(x) if isinstance(x, dict | list) else str(x)


def _load_domain_config(domain: str) -> tuple[GraphRagConfig, Path]:
    """Load and resolve configuration for a given domain.

    Args:
        domain (str): Domain directory name.

    Returns:
        tuple: (domain_config, output_dir)
    """
    domain_root = Path(f"{app_config.GRAPHRAG_ROOT_DIR}/{domain}").resolve()
    domain_config = load_config(domain_root)
    output_dir = Path(domain_config.output.base_dir)
    return domain_config, output_dir


def _load_parquets(output_dir: Path, wants: list[str]) -> dict[str, pd.DataFrame]:
    parquets = {
        "communities": "communities.parquet",
        "community_reports": "community_reports.parquet",
        "text_units": "text_units.parquet",
        "relationships": "relationships.parquet",
        "entities": "entities.parquet",
        "covariates": "covariates.parquet",
    }
    data = DictDotNotation()
    for key in wants:
        data[key] = _read_parquet(output_dir / parquets[key])
    return data


def _load_search_data(domain: str, indexes: list[str]) -> tuple[GraphRagConfig, Path, dict[str, pd.DataFrame]]:
    """Load common Parquet files.

    Returns:
        tuple: (domain_config, output_dir, data dict)
    """
    config, output_dir = _load_domain_config(domain)
    data = _load_parquets(output_dir, wants=indexes)
    return config, output_dir, data


def _load_description_embedding_store(output_dir: Path) -> LanceDBVectorStore:
    """Load the description embedding store.

    Args:
        output_dir (Path): Output directory from configuration.

    Returns:
        LanceDBVectorStore: Initialized embedding store.
    """
    store = LanceDBVectorStore(collection_name="default-entity-description")
    store.connect(db_uri=output_dir / "lancedb")
    return store


def _load_full_content_embedding_store(output_dir: Path) -> LanceDBVectorStore:
    """Load the full-content embedding store.

    Args:
        output_dir (Path): Output directory from configuration.

    Returns:
        LanceDBVectorStore: Initialized full-content store.
    """
    store = LanceDBVectorStore(collection_name="default-community-full_content")
    store.connect(db_uri=output_dir / "lancedb")
    return store

# -----------------------------------------------------------------------------
# Endpoints: GET Endpoints (using helper functions)
# -----------------------------------------------------------------------------

@app.get("/v1/search/{domain}/local")
async def simple_local_search_v1(
    domain: str,
    query: Annotated[str, Query(description="Search query for local context")] = ...,
    community_level: Annotated[int, Query(
        description=("Community level in the Leiden hierarchy; higher values represent smaller communities.")
    )] = 2,
    response_type: Annotated[str, Query(description="The type of response to return")] = "Single paragraph",
) -> SearchResponse:
    """Perform a local context search for a specified domain (GET)."""
    try:
        config, output_dir, data = _load_search_data(domain, [
            "communities", "community_reports", "text_units", "relationships", "entities", "covariates"
        ])
        community_reports = read_indexer_reports(data.community_reports, data.communities, community_level)
        text_units = read_indexer_text_units(data.text_units)
        relationships = read_indexer_relationships(data.relationships)
        entities = read_indexer_entities(data.entities, data.communities, community_level)
        claims = read_indexer_covariates(data.covariates) if data.covariates is not None else []

        description_embedding_store = _load_description_embedding_store(output_dir)

        search_engine = get_local_search_engine(
            config=config,
            reports=community_reports,
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            covariates={"claims": claims},
            response_type=response_type,
            description_embedding_store=description_embedding_store,
        )
        result = await search_engine.search(query)
        return SearchResponse(result=_tostring(result.response), query=query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/v1/search/{domain}/global")
async def simple_global_search_v1(
    domain: str,
    query: Annotated[str, Query(description="Search query for global context")] = ...,
    community_level: Annotated[int, Query(
        description=("Community level in the Leiden hierarchy; higher values represent smaller communities.")
    )] = 2,
    response_type: Annotated[str, Query(description="The type of response to return")] = "Single paragraph",
) -> SearchResponse:
    """Perform a global context search for a specified domain (GET)."""
    try:
        config, _, data = _load_search_data(domain, ["entities", "communities", "community_reports"])
        entities = read_indexer_entities(data.entities, data.communities, community_level)
        community_reports = read_indexer_reports(data.community_reports, data.communities, community_level)
        communities = read_indexer_communities(data.communities, data.community_reports)

        search_engine = get_global_search_engine(
            config=config,
            reports=community_reports,
            entities=entities,
            communities=communities,
            response_type=response_type,
        )
        result = await search_engine.search(query)
        return SearchResponse(result=_tostring(result.response), query=query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/v1/search/{domain}/drift")
async def simple_drift_search_v1(
    domain: str,
    query: Annotated[str, Query(description="Search query for drift context")] = ...,
    community_level: Annotated[int, Query(
        description=("Community level in the Leiden hierarchy; higher values represent smaller communities.")
    )] = 2,
    response_type: Annotated[str, Query(description="The type of response to return")] = "Single paragraph",
) -> SearchResponse:
    """Perform a drift context search for a specified domain (GET)."""
    try:
        config, output_dir, data = _load_search_data(domain, [
            "communities", "community_reports", "text_units", "relationships", "entities"
        ])
        entities = read_indexer_entities(data.entities, data.communities, community_level)
        relationships = read_indexer_relationships(data.relationships)
        community_reports = read_indexer_reports(data.community_reports, data.communities, community_level)
        text_units = read_indexer_text_units(data.text_units)

        description_embedding_store = _load_description_embedding_store(output_dir)
        full_content_embedding_store = _load_full_content_embedding_store(output_dir)
        # Populate full-content embeddings.
        read_indexer_report_embeddings(community_reports, full_content_embedding_store)

        search_engine = get_drift_search_engine(
            config=config,
            reports=community_reports,
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            description_embedding_store=description_embedding_store,
            response_type=response_type,
        )
        result = await search_engine.search(query=query)
        return SearchResponse(result=_tostring(result.response), query=query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/v1/search/{domain}/basic")
async def simple_basic_search_v1(
    domain: str,
    query: Annotated[str, Query(description="Search query for basic context")] = ...,
) -> SearchResponse:
    """Perform a basic context search for a specified domain (GET)."""
    try:
        domain_config, output_dir, data = _load_search_data(domain, ["text_units"])
        text_units = read_indexer_text_units(data.text_units)

        description_embedding_store = _load_description_embedding_store(output_dir)

        search_engine = get_basic_search_engine(
            config=domain_config,
            text_units=text_units,
            text_unit_embeddings=description_embedding_store,
        )
        result = await search_engine.search(query=query)
        return SearchResponse(result=_tostring(result.response), query=query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/")
async def status() -> str:
    """Return the API status."""
    return "GraphRAG API is up and running"

# -----------------------------------------------------------------------------
# POST Endpoints (using the same helper functions)
# -----------------------------------------------------------------------------

@app.post("/v1/search/{domain}/local")
async def local_search_v1(domain: str, req: SearchRequest) -> SearchResponse:
    """Perform a local context search (POST version)."""
    try:
        config, output_dir, data = _load_search_data(domain, [
            "communities", "community_reports", "text_units", "relationships", "entities", "covariates"
        ])
        community_reports = read_indexer_reports(data.community_reports, data.communities, req.community_level)
        text_units = read_indexer_text_units(data.text_units)
        relationships = read_indexer_relationships(data.relationships)
        entities = read_indexer_entities(data.entities, data.communities, req.community_level)
        claims = read_indexer_covariates(data.covariates) if data.covariates is not None else []

        description_embedding_store = _load_description_embedding_store(output_dir)

        search_engine = get_local_search_engine(
            config=config,
            reports=community_reports,
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            covariates={"claims": claims},
            response_type=req.response_type,
            description_embedding_store=description_embedding_store,
        )
        result = await search_engine.search(req.query)
        return SearchResponse(result=_tostring(result.response), query=req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/search/{domain}/global")
async def global_search_v1(domain: str, req: SearchRequest) -> SearchResponse:
    """Perform a global context search (POST version)."""
    try:
        config, _, data = _load_search_data(domain, ["entities", "communities", "community_reports"])
        entities = read_indexer_entities(data.entities, data.communities, req.community_level)
        community_reports = read_indexer_reports(data.community_reports, data.communities, req.community_level)
        communities = read_indexer_communities(data.communities, data.community_reports)

        search_engine = get_global_search_engine(
            config=config,
            reports=community_reports,
            entities=entities,
            communities=communities,
            response_type=req.response_type,
        )
        result = await search_engine.search(req.query)
        return SearchResponse(result=_tostring(result.response), query=req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/search/{domain}/drift")
async def drift_search_v1(domain: str, req: SearchRequest) -> SearchResponse:
    """Perform a drift context search (POST version)."""
    try:
        config, output_dir, data = _load_search_data(domain, [
            "communities", "community_reports", "text_units", "relationships", "entities"
        ])
        entities = read_indexer_entities(data.entities, data.communities, req.community_level)
        relationships = read_indexer_relationships(data.relationships)
        community_reports = read_indexer_reports(data.community_reports, data.communities, req.community_level)
        text_units = read_indexer_text_units(data.text_units)

        description_embedding_store = _load_description_embedding_store(output_dir)
        full_content_embedding_store = _load_full_content_embedding_store(output_dir)
        # Populate full-content embeddings.
        read_indexer_report_embeddings(community_reports, full_content_embedding_store)

        search_engine = get_drift_search_engine(
            config=config,
            reports=community_reports,
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            description_embedding_store=description_embedding_store,
            response_type=req.response_type,
        )
        result = await search_engine.search(query=req.query)
        return SearchResponse(result=_tostring(result.response), query=req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/search/{domain}/basic")
async def basic_search_v1(domain: str, req: SearchRequest) -> SearchResponse:
    """Perform a basic context search (POST version)."""
    try:
        config, output_dir, data = _load_search_data(domain, ["text_units"])
        text_units = read_indexer_text_units(data.text_units)

        description_embedding_store = _load_description_embedding_store(output_dir)

        search_engine = get_basic_search_engine(
            config=config,
            text_units=text_units,
            text_unit_embeddings=description_embedding_store,
        )
        result = await search_engine.search(query=req.query)
        return SearchResponse(result=_tostring(result.response), query=req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def run() -> None:
    """Run the GraphRAG API service using uvicorn.

    The application starts on host 0.0.0.0 and the port specified in the configuration.
    """
    uvicorn.run(app, host="0.0.0.0", port=app_config.GRAPHRAG_API_PORT)
