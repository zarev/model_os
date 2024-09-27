"""Caching mechanism for model loading"""

from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

class ModelCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            oldest = next(iter(self.cache))
            del self.cache[oldest]
            logger.info(f"Evicted model {oldest} from cache")
        self.cache[key] = value
        logger.info(f"Added model {key} to cache")

    def clear(self):
        self.cache.clear()
        logger.info("Cleared model cache")

model_cache = ModelCache(capacity=2)  # Adjust capacity as needed
