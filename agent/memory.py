import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class EpisodicMemory(BaseModel):
    run_id: str
    step_index: int
    content: str
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SemanticMemory(BaseModel):
    key: str
    content: str
    confidence: float = 1.0
    source_run_id: str
    ttl: Optional[float] = None # Time-to-live timestamp or duration
    created_at: float = Field(default_factory=time.time)

class MemoryStore:
    """
    Memory Store managing Working Memory (key-value scratchpad), 
    Episodic Memory (run trajectories and historical steps), 
    and Semantic Memory (curated knowledge with confidence, TTL, and gating).
    """
    def __init__(self):
        self.working_memory: Dict[str, Any] = {}
        self.episodic_memory: List[EpisodicMemory] = []
        self.semantic_memory: Dict[str, SemanticMemory] = {}

    # Working Memory Operations
    def working_set(self, key: str, value: Any):
        self.working_memory[key] = value

    def working_get(self, key: str, default: Any = None) -> Any:
        return self.working_memory.get(key, default)

    # Episodic Memory Operations
    def record_episode(self, run_id: str, step_index: int, content: str, metadata: Optional[Dict[str, Any]] = None):
        episode = EpisodicMemory(
            run_id=run_id,
            step_index=step_index,
            content=content,
            metadata=metadata or {}
        )
        self.episodic_memory.append(episode)

    def retrieve_episodes(self, run_id: Optional[str] = None, limit: int = 10) -> List[EpisodicMemory]:
        episodes = self.episodic_memory
        if run_id:
            episodes = [ep for ep in episodes if ep.run_id == run_id]
        return episodes[-limit:]

    # Semantic Memory Operations
    def write_semantic(
        self,
        key: str,
        content: str,
        source_run_id: str,
        confidence: float = 1.0,
        min_confidence_gate: float = 0.5,
        ttl_seconds: Optional[float] = None
    ) -> bool:
        """Writes to semantic memory if confidence passes the gating threshold."""
        if confidence < min_confidence_gate:
            return False

        expiry = time.time() + ttl_seconds if ttl_seconds is not None else None
        self.semantic_memory[key] = SemanticMemory(
            key=key,
            content=content,
            confidence=confidence,
            source_run_id=source_run_id,
            ttl=expiry
        )
        return True

    def retrieve_semantic(self, query: str) -> List[SemanticMemory]:
        """Scoped retrieval of semantic memories matching query, filtering out expired items."""
        now = time.time()
        results = []
        for mem in self.semantic_memory.values():
            if mem.ttl and mem.ttl < now:
                continue
            if query.lower() in mem.key.lower() or query.lower() in mem.content.lower():
                results.append(mem)
        return results
