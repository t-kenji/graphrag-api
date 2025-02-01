"""Microsoft GraphRAG web api package."""

import json
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import pandas as pd
import tiktoken
import tomllib
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from graphrag.config.models.drift_search_config import DRIFTSearchConfig
from graphrag.logger.print_progress import PrintProgressLogger
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_communities,
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_report_embeddings,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.basic_search.basic_context import BasicSearchContext
from graphrag.query.structured_search.basic_search.search import BasicSearch
from graphrag.query.structured_search.drift_search.drift_context import DRIFTSearchContextBuilder
from graphrag.query.structured_search.drift_search.search import DRIFTSearch
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore

__version__ = version(__package__)

logger = PrintProgressLogger("")

class DictDotNotation(dict):
    """You can access the dictionary with Dot Notation."""
    def __init__(self, *args: list | None, **kwargs: dict | None) -> None:
        """Initializer."""
        super().__init__(*args, **kwargs)
        self.__dict__ = self

config = DictDotNotation({
    "GRAPHRAG_API_PORT": 3080,
    "GRAPHRAG_API_KEY": "<YOUR API KEY>",
    "GRAPHRAG_LLM_API_BASE": "https://api.oepnai.com/v1",
    "GRAPHRAG_LLM_MODEL": "gpt-4o",
    "GRAPHRAG_EMBEDDING_API_BASE": "https://api.openai.com/v1",
    "GRAPHRAG_EMBEDDING_MODEL": "text-embedding-3-large",
    "GRAPHRAG_ROOT_DIR": ".",
})

config_file = Path("environ.toml")
if config_file.exists():
    with config_file.open("rb") as f:
        for k, v in tomllib.load(f).items():
            if k in config:
                config[k] = v

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"http://localhost:{config.GRAPHRAG_API_PORT}"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatOpenAI(
    api_key=config.GRAPHRAG_API_KEY,
    api_base=config.GRAPHRAG_LLM_API_BASE,
    model=config.GRAPHRAG_LLM_MODEL,
    api_type=OpenaiApiType.OpenAI,
    max_retries=5,
    request_timeout=300.0,
)
text_embeder = OpenAIEmbedding(
    api_key=config.GRAPHRAG_API_KEY,
    api_base=config.GRAPHRAG_EMBEDDING_API_BASE,
    model=config.GRAPHRAG_EMBEDDING_MODEL,
    api_type=OpenaiApiType.OpenAI,
    max_retries=5,
    request_timeout=300.0,
)
token_encoder = tiktoken.get_encoding("cl100k_base")


def read_parquet(path: Path) -> pd.DataFrame | None:
    """Read parquet data frame file."""
    return pd.read_parquet(path) if path.is_file() else None


def tostring(x: any) -> str:
    """Response data to string."""
    return json.dumps(x) if isinstance(x, dict | list) else str(x)


@app.get("/v1/search/local/{domain}")
async def local_search_v1(
    domain: str, query: Annotated[str, Query(description="Search query for local context")] = ...,
) -> Response:
    """Search query for local context."""
    try:
        root_path = Path(f"{config.GRAPHRAG_ROOT_DIR}/{domain}").resolve()
        final_nodes = read_parquet(root_path / "output/create_final_nodes.parquet") # TODO: Memoizeする
        final_entities = read_parquet(root_path / "output/create_final_entities.parquet") # TODO: Memoizeする
        final_community_reports = read_parquet(root_path / "output/create_final_community_reports.parquet") # TODO: Memoizeする
        final_relationships = read_parquet(root_path / "output/create_final_relationships.parquet") # TODO: Memoizeする
        final_covariates = read_parquet(root_path / "output/create_final_covariates.parquet") # TODO: Memoizeする
        final_text_units = read_parquet(root_path / "output/create_final_text_units.parquet") # TODO: メモライスする
        entities = read_indexer_entities(final_nodes, final_entities, 2) # TODO: Memoizeする
        relationships = read_indexer_relationships(final_relationships) # TODO: メモライスする
        claims = read_indexer_covariates(final_covariates) if final_covariates is not None else [] # TODO: Memoizeする
        community_reports = read_indexer_reports(final_community_reports, final_nodes, 2) # TODO: Memoizeする
        text_units = read_indexer_text_units(final_text_units) # TODO: Memoizeする
        description_embedding_store = LanceDBVectorStore(collection_name="default-entity-description") # TODO: Memoizeする
        description_embedding_store.connect(db_uri=root_path / "output/lancedb") # TODO: Memoizeする
        context_builder = LocalSearchMixedContext( # TODO: Memoizeする
            entities=entities,
            entity_text_embeddings=description_embedding_store,
            text_embedder=text_embeder,
            text_units=text_units,
            community_reports=community_reports,
            relationships=relationships,
            covariates={"claims": claims},
            token_encoder=token_encoder,
            embedding_vectorstore_key=EntityVectorStoreKey.ID,
        )
        searcher = LocalSearch(
            llm=llm,
            context_builder=context_builder,
            token_encoder=token_encoder,
            llm_params={
                "max_tokens": 512,
                "temperature": 0.0,
            },
            context_builder_params={
                "text_unit_prop": 0.5,
                "community_prop": 0.1,
                "conversation_history_max_turns": 5,
                "conversation_history_user_turns_only": True,
                "top_k_mapped_entities": 10,
                "top_k_relationships": 10,
                "include_entity_rank": True,
                "include_relationship_weight": True,
                "include_community_rank": False,
                "return_candidate_context": False,
                "embedding_vectorstore_key": EntityVectorStoreKey.ID,
                "max_tokens": 512,
            },
            response_type="single paragraph",
        )
        result = await searcher.asearch(query)
        return JSONResponse(
            content={
                "result": tostring(result.response),
                "query": query,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/v1/search/global/{domain}")
async def global_search_v1(
    domain: str, query: Annotated[str, Query(description="Search query for global context")] = ...,
) -> Response:
    """Search query for global context."""
    try:
        root_path = Path(f"{config.GRAPHRAG_ROOT_DIR}/{domain}").resolve()
        final_nodes = read_parquet(root_path / "output/create_final_nodes.parquet") # TODO: Memoizeする
        final_entities = read_parquet(root_path / "output/create_final_entities.parquet") # TODO: Memoizeする
        final_community_reports = read_parquet(root_path / "output/create_final_community_reports.parquet") # TODO: Memoizeする
        final_communities = read_parquet(root_path / "output/create_final_communities.parquet") # TODO: Memoizeする
        entities = read_indexer_entities(final_nodes, final_entities, 2) # TODO: Memoizeする
        community_reports = read_indexer_reports(final_community_reports, final_nodes, 2) # TODO: Memoizeする
        communities = read_indexer_communities(final_communities, final_nodes, final_community_reports) # TODO: Memoizeする
        context_builder = GlobalCommunityContext( # TODO: Memoizeする
            community_reports=community_reports,
            communities=communities,
            entities=entities,
            token_encoder=token_encoder,
        )
        searcher = GlobalSearch(
            llm=llm,
            context_builder=context_builder,
            token_encoder=token_encoder,
            response_type="single paragraph",
            allow_general_knowledge=False,
            json_mode=True,
            max_data_tokens=12_000,
            map_llm_params={
                "max_tokens": 512,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
            reduce_llm_params={
                "max_tokens": 1024,
                "temperature": 0.0,
            },
            context_builder_params={
                "use_community_summary": False,
                "shuffle_data": True,
                "include_community_rank": True,
                "min_community_rank": 0,
                "community_rank_name": "rank",
                "include_community_weight": True,
                "community_weight_name": "occurrence weight",
                "normalize_community_weight": True,
                "max_tokens": 2048,
                "context_name": "Reports",
            },
            concurrent_coroutines=2,
        )
        result = await searcher.asearch(query)
        return JSONResponse(content={"result": tostring(result.response), "query": query})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/v1/search/drift/{domain}")
async def drift_search_v1(
    domain: str, query: Annotated[str, Query(description="Search query for drift context")] = ...,
) -> Response:
    """Search query for drift context."""
    try:
        root_path = Path(f"{config.GRAPHRAG_ROOT_DIR}/{domain}").resolve()
        final_nodes = read_parquet(root_path / "output/create_final_nodes.parquet") # TODO: Memoizeする
        final_entities = read_parquet(root_path / "output/create_final_entities.parquet") # TODO: Memoizeする
        final_relationships = read_parquet(root_path / "output/create_final_relationships.parquet") # TODO: Memoizeする
        final_community_reports = read_parquet(root_path / "output/create_final_community_reports.parquet") # TODO: Memoizeする
        final_text_units = read_parquet(root_path / "output/create_final_text_units.parquet") # TODO: メモライスする
        entities = read_indexer_entities(final_nodes, final_entities, 2) # TODO: Memoizeする
        relationships = read_indexer_relationships(final_relationships) # TODO: メモライスする
        community_reports = read_indexer_reports(final_community_reports, final_nodes, 2) # TODO: Memoizeする
        text_units = read_indexer_text_units(final_text_units) # TODO: Memoizeする
        description_embedding_store = LanceDBVectorStore(collection_name="default-entity-description") # TODO: Memoizeする
        description_embedding_store.connect(db_uri=root_path / "output/lancedb") # TODO: Memoizeする
        full_content_embedding_store = LanceDBVectorStore(collection_name="default-community-full_content")
        full_content_embedding_store.connect(db_uri=root_path / "output/lancedb")
        read_indexer_report_embeddings(community_reports, full_content_embedding_store)
        drift_config = DRIFTSearchConfig(
            temperature=0,
            max_tokens=12_000,
            primer_folds=1,
            drift_k_followups=3,
            n_depth=3,
            n=1,
            concurrency=1,
        )
        engine = DRIFTSearch(
            llm=llm,
            context_builder=DRIFTSearchContextBuilder(
                chat_llm=llm,
                text_embedder=text_embeder,
                entities=entities,
                reports=community_reports,
                text_units=text_units,
                relationships=relationships,
                token_encoder=token_encoder,
                config=drift_config,
                entity_text_embeddings=description_embedding_store,
                response_type="single paragraph",
            ),
            token_encoder=token_encoder,
        )
        result = await engine.asearch(query=query)
        return JSONResponse(content={"result": tostring(result.response), "query": query})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/v1/search/basic/{domain}")
async def basic_search_v1(
    domain: str, query: Annotated[str, Query(description="Search query for basic context")] = ...,
) -> Response:
    """Search query for basic context."""
    try:
        root_path = Path(f"{config.GRAPHRAG_ROOT_DIR}/{domain}").resolve()
        final_nodes = read_parquet(root_path / "output/create_final_nodes.parquet") # TODO: Memoizeする
        final_community_reports = read_parquet(root_path / "output/create_final_community_reports.parquet") # TODO: Memoizeする
        final_text_units = read_parquet(root_path / "output/create_final_text_units.parquet") # TODO: メモライスする
        community_reports = read_indexer_reports(final_community_reports, final_nodes, 2) # TODO: Memoizeする
        text_units = read_indexer_text_units(final_text_units) # TODO: Memoizeする
        description_embedding_store = LanceDBVectorStore(collection_name="default-entity-description") # TODO: Memoizeする
        description_embedding_store.connect(db_uri=root_path / "output/lancedb") # TODO: Memoizeする
        full_content_embedding_store = LanceDBVectorStore(collection_name="default-community-full_content")
        full_content_embedding_store.connect(db_uri=root_path / "output/lancedb")
        text_unit_text_embedding_store = LanceDBVectorStore(collection_name="default-text_unit-text")
        text_unit_text_embedding_store.connect(db_uri=root_path / "output/lancedb")
        read_indexer_report_embeddings(community_reports, full_content_embedding_store)
        engine = BasicSearch(
            llm=llm,
            context_builder=BasicSearchContext(
                text_embedder=text_embeder,
                text_unit_embeddings=text_unit_text_embedding_store,
                text_units=text_units,
                token_encoder=token_encoder,
            ),
            token_encoder=token_encoder,
        )
        result = await engine.asearch(query=query)
        return JSONResponse(content={"result": tostring(result.response), "query": query})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/")
async def status() -> str:
    """Response status."""
    return "GraphRAG API is up and running"


def main() -> None:
    """Execute the main program."""
    uvicorn.run(app, host="0.0.0.0", port=config.GRAPHRAG_API_PORT)


if __name__ == "__main__":
    main()
