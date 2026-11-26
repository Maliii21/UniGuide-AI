
"""
Thread-safe shared retrieval resources for UniGuide-AI.
"""

from threading import Lock
from copy import deepcopy

from .retrieval import RetrievalEngine
from .relevance_filter import DEFAULT_THRESHOLD


class SharedRetrievalEngine:
    """Safely share retrieval resources across sessions."""

    def __init__(
        self,
        device="cpu",
        retrieval_engine=None
    ):

        self._engine = (
            retrieval_engine
            if retrieval_engine is not None
            else RetrievalEngine(device=device)
        )

        self._lock = Lock()

    def retrieve_with_relevance(
        self,
        query,
        top_k=5,
        candidate_k=20,
        rrf_k=60,
        threshold=DEFAULT_THRESHOLD
    ):
        """Execute retrieval with exclusive resource access."""

        with self._lock:

            result = self._engine.retrieve_with_relevance(
                query=query,
                top_k=top_k,
                candidate_k=candidate_k,
                rrf_k=rrf_k,
                threshold=threshold
            )

            return deepcopy(result)
