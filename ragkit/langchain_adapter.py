"""Optional real LangChain retriever; engine lifecycle stays with the caller."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from pydantic import ConfigDict, Field
except ImportError as exc:
    raise RuntimeError('LangChain integration requires langchain-core; install RagKit with its langchain extra.') from exc


class RagKitRetriever(BaseRetriever):
    """Adapt a RagKit engine to LangChain's invoke/ainvoke retrieval interface.

    ``engine`` is an already-open RagKit instance; this adapter does not close
    it or build a dense index. Missing/stale index failures remain explicit.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    engine: Any = Field(exclude=True, repr=False)
    top_k: int = Field(default=5, ge=1, le=20, strict=True)
    mode: Literal['lexical', 'dense', 'hybrid'] = 'lexical'
    rerank: bool = False
    document_ids: list[str] | None = None

    def _get_relevant_documents(self, query: str, *,
                                run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        result = self.engine.search(query, top_k=self.top_k, mode=self.mode,
                                    rerank=self.rerank, document_ids=self.document_ids)
        return [Document(page_content=row['text'], id=row['chunk_id'],
                         metadata=deepcopy({key: value for key, value in row.items() if key != 'text'}))
                for row in result['results']]
