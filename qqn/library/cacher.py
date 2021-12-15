class Cacher:

    def __init__(self, cache_max_size=None):
        super()
        self.cache_accesses = dict()
        self.cache = dict()
        self.cache_size = 0
        self.cache_max_size = cache_max_size
        self.least_accessed_trace = None
        self.least_accessed_accesses = 0

    def learn(self, *args):
        assert args, "Arguments of learn may not be empty."
        pointer = 0
        cache = self.cache
        cache_accesses = self.cache_accesses
        curr_arg = args[pointer]
        curr_key = self._arg_to_key(curr_arg)
        while len(args) < (pointer + 1):
            if curr_key not in cache:
                cache[curr_key] = {}
                cache_accesses[curr_key] = {}
            cache = cache[curr_key]
            cache_accesses = cache_accesses[curr_key]
            pointer += 1
            curr_arg = args[pointer]
            curr_key = self._arg_to_key(curr_arg)

        if curr_key not in cache:
            if self.__max_cache_size() <= self.cache_size + 1:
                self.__demote_least_accessed()
                self.cache_size -= 1
            cache_accesses[curr_key] = 0
            cache[curr_key] = self._learn(*args)
            self.cache_size += 1
            self.least_accessed_accesses = 1
            self.least_accessed_trace = args

        cache_accesses[curr_key] += 1

        return cache[curr_key]

    def _arg_to_key(self, arg):
        return hash(arg)

    def _learn(self, *args):
        raise NotImplementedError

    def __max_cache_size(self):
        return self.cache_max_size or (self.cache_size + 2)

    def __demote_least_accessed(self):
        pass
