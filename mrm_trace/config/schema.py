from __future__ import annotations

from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field, field_validator, model_validator


class EngineType(str, Enum):
    LLAMACPP = "llamacpp"
    VLLM = "vllm"


class CollectorType(str, Enum):
    PERF_MEM = "perf_mem"
    MEMRAY = "memray"
    PROCESS_MONITOR = "process_monitor"


class UnknownPolicy(str, Enum):
    LOG_AND_SKIP = "log_and_skip"
    ASSIGN_RUNTIME = "assign_runtime"
    ASSIGN_UNKNOWN = "assign_unknown"
    FAIL_RUN = "fail_run"


class MetricName(str, Enum):
    RETENTION_DURATION = "retention_duration"
    WRITE_ONCE_RATIO = "write_once_ratio"
    READ_FREQUENCY = "read_frequency"
    WORKING_SET_SIZE = "working_set_size"
    ACCESS_LOCALITY = "access_locality"
    INTER_ACCESS_INTERVAL = "inter_access_interval"


class Compression(str, Enum):
    ZSTD = "zstd"
    SNAPPY = "snappy"
    GZIP = "gzip"
    NONE = "none"


class ExperimentMeta(BaseModel):
    name: str
    description: str = ""
    version: str = "1.0.0"
    date_created: str = ""
    authors: List[str] = Field(default_factory=list)
    institution: str = ""
    paper_ref: str = ""


class ModelConfig(BaseModel):
    id: str
    engine: EngineType
    path: Optional[str] = None
    hf_repo: Optional[str] = None
    family: str = ""
    params_billions: Optional[float] = None
    quantisation: Optional[str] = None
    dtype: str = "float16"

    @model_validator(mode="after")
    def check_path_or_repo(self) -> "ModelConfig":
        if self.engine == EngineType.LLAMACPP and not self.path:
            raise ValueError(
                f"Model '{self.id}' uses llamacpp but has no 'path' defined"
            )
        if self.engine == EngineType.VLLM and not self.hf_repo and not self.path:
            raise ValueError(
                f"Model '{self.id}' uses vllm but has neither 'path' nor 'hf_repo'"
            )
        return self


class SweepConfig(BaseModel):
    context_lengths: List[int] = Field(default_factory=lambda: [2048])
    batch_sizes: List[int] = Field(default_factory=lambda: [1])
    output_tokens: List[int] = Field(default_factory=lambda: [128])
    repetitions: int = 1
    warmup_runs: int = 0

    @field_validator("context_lengths", "batch_sizes", "output_tokens")
    @classmethod
    def positive_values(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("Must have at least one value")
        if any(x <= 0 for x in v):
            raise ValueError("All values must be positive integers")
        return v

    @field_validator("repetitions")
    @classmethod
    def positive_repetitions(cls, v: int) -> int:
        if v < 1:
            raise ValueError("repetitions must be >= 1")
        return v

    @field_validator("warmup_runs")
    @classmethod
    def non_negative_warmup(cls, v: int) -> int:
        if v < 0:
            raise ValueError("warmup_runs must be >= 0")
        return v


class PromptsConfig(BaseModel):
    source: str
    categories: List[str] = Field(default_factory=list)


class PerfMemConfig(BaseModel):
    sample_period: int = 4000
    ldlat_threshold: int = 20
    events: List[str] = Field(
        default_factory=lambda: ["cpu/mem-loads,ldlat=20/P", "cpu/mem-stores/P"]
    )
    extra_flags: List[str] = Field(default_factory=list)
    require_root: bool = True


class MemrayConfig(BaseModel):
    native_traces: bool = True
    follow_fork: bool = True
    file_size_limit_gb: int = 20


class ProcessMonitorConfig(BaseModel):
    poll_interval_ms: int = 100
    metrics: List[str] = Field(
        default_factory=lambda: ["rss", "vms", "shared", "text", "data"]
    )


class CollectorConfig(BaseModel):
    primary: CollectorType = CollectorType.PERF_MEM
    fallback: Optional[CollectorType] = CollectorType.MEMRAY
    fallback_2: Optional[CollectorType] = CollectorType.PROCESS_MONITOR
    perf_mem: PerfMemConfig = Field(default_factory=PerfMemConfig)
    memray: MemrayConfig = Field(default_factory=MemrayConfig)
    process_monitor: ProcessMonitorConfig = Field(default_factory=ProcessMonitorConfig)


class TelemetryConfig(BaseModel):
    capture: bool = True
    thermal_sampling_interval_s: int = 5
    thermal_source: str = "auto"
    invalidate_on_throttle: bool = True
    software_versions: bool = True
    hardware_topology: bool = True
    run_fio_benchmark: bool = False


class LabellerConfig(BaseModel):
    rules_file: str = "config/models.yaml"
    unknown_policy: UnknownPolicy = UnknownPolicy.LOG_AND_SKIP
    min_sample_count: int = 10


class AnalyserConfig(BaseModel):
    metrics: List[MetricName] = Field(
        default_factory=lambda: [
            MetricName.RETENTION_DURATION,
            MetricName.WRITE_ONCE_RATIO,
            MetricName.READ_FREQUENCY,
            MetricName.WORKING_SET_SIZE,
        ]
    )
    histogram_bins: int = 100
    cdf_resolution: int = 1000
    group_by: List[str] = Field(
        default_factory=lambda: ["region_type", "context_length"]
    )


class OutputFormats(BaseModel):
    raw_traces: str = "parquet"
    metrics: str = "csv"
    figures: str = "pdf"
    metadata: str = "json"


class OutputConfig(BaseModel):
    base_dir: str = "results/"
    formats: OutputFormats = Field(default_factory=OutputFormats)
    compression: Compression = Compression.ZSTD
    hf_dataset_push: bool = False
    hf_repo: Optional[str] = None


class ExperimentConfig(BaseModel):
    experiment: ExperimentMeta
    models: List[ModelConfig]
    sweep: SweepConfig = Field(default_factory=SweepConfig)
    prompts: Optional[PromptsConfig] = None
    collector: CollectorConfig = Field(default_factory=CollectorConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    labeller: LabellerConfig = Field(default_factory=LabellerConfig)
    analyser: AnalyserConfig = Field(default_factory=AnalyserConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("models")
    @classmethod
    def at_least_one_model(cls, v: List[ModelConfig]) -> List[ModelConfig]:
        if not v:
            raise ValueError("At least one model must be specified")
        return v


class RunConfig(BaseModel):
    """A single materialised run from the sweep expansion."""

    run_id: str
    model: ModelConfig
    context_length: int
    batch_size: int
    output_tokens: int
    rep_index: int
    is_warmup: bool = False
    output_dir: Optional[str] = None
    # Generation stops at the model's end-of-turn token, so output_tokens is a
    # *ceiling*, not a count: asked for 1977, Llama-3.2-1B produced 653 and the
    # KV cache reached 33% occupancy instead of 98%.  An occupancy experiment
    # needs the token count to be the independent variable, which means
    # suppressing that token.  Off by default -- it makes the model babble past
    # its natural stopping point, which is right for a memory measurement and
    # wrong for anything that reads the text.
    ignore_eos: bool = False
