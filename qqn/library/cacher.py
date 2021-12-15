from typing import Optional

from cachetools import Cache, LFUCache, FIFOCache, LRUCache, MRUCache, RRCache

from qqn.library.common import to_key
from qqn.library.effect import Messenger


class Cacher(Messenger):

    def __init__(self, types='__ALL', ignore_types=None, maxsize=8192, method='LFU', cache: Optional[Cache] = None):
        super().__init__()

        self.types = types
        self.ignore_types = ignore_types
        self.cache = cache
        if self.cache is None:
            if method == 'FIFO':
                self.cache = FIFOCache(maxsize)
            elif method == 'LRU':
                self.cache = LRUCache(maxsize)
            elif method == 'MRU':
                self.cache = MRUCache(maxsize)
            elif method == 'RR':
                self.cache = RRCache(maxsize)
            else:
                self.cache = LFUCache(maxsize)

    def process_message(self, msg):
        if not msg['done'] and self.types is not None and (self.types == '_ALL' or msg['type'] in self.types) and (
                self.ignore_types is None or msg['type'] not in self.ignore_types):
            key = self._msg_to_key(msg)
            if key in self.cache:
                msg['value'] = self.cache[key]
                msg['done'] = True

    def postprocess_message(self, msg):
        if msg['done'] and self.types is not None and (self.types == '_ALL' or msg['type'] in self.types) and (
                self.ignore_types is None or msg['type'] not in self.ignore_types):
            key = self._msg_to_key(msg)
            if key not in self.cache:
                self.cache[key] = msg['value']
            elif self.cache[key] != msg['value']:
                self.cache[key] = msg['value']

    def _msg_to_key(self, msg):
        key = hash(msg['type'])
        key = self._args_to_key(msg['args'], key)
        key = self._kwargs_to_key(msg['kwargs'], key)
        return key

    def _args_to_key(self, args, key=0):
        for arg in args:
            key = hash((key, to_key(arg)))
        return key

    def _kwargs_to_key(self, kwargs, key=0):
        for k, v in sorted(kwargs.items()):
            key = hash((key, (k, v)))
        return key
