from functools import lru_cache
import hashlib
import json
from typing import Dict, Any, Optional

class SearchCache:
    def __init__(self, max_size: int = 100):
        self._cache = {}
        self.max_size = max_size

    def get_query_hash(self, query_string_or_tensor) -> str:
        """Generate a hash for either text query or image tensor"""
        if isinstance(query_string_or_tensor, str):
            return hashlib.md5(query_string_or_tensor.encode()).hexdigest()
        return hashlib.md5(str(query_string_or_tensor.tolist()).encode()).hexdigest()

    def get(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve results from cache"""
        return self._cache.get(query_hash)

    def set(self, query_hash: str, results: Dict[str, Any]):
        """Store results in cache"""
        if len(self._cache) >= self.max_size:
            # Remove oldest entry if cache is full
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[query_hash] = results

    def clear(self):
        """Clear the cache"""
        self._cache.clear()