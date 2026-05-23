from enum import Enum


class RegionType(str, Enum):
    WEIGHT = "weight"
    KV_CACHE = "kv_cache"
    ACTIVATION = "activation"
    RUNTIME = "runtime"
    UNKNOWN = "unknown"


class OpType(str, Enum):
    LOAD = "load"
    STORE = "store"


class Phase(str, Enum):
    WEIGHT_LOAD = "weight_load"
    GENERATION = "generation"
    TEARDOWN = "teardown"


class RunValidity(str, Enum):
    CLEAN = "clean"
    MARGINAL = "marginal"
    CONTAMINATED = "contaminated"


class RetentionClass(str, Enum):
    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"
    SESSION = "session"
