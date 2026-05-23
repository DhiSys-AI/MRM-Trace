# mrm-trace

mrm-trace is a Python project to collect and analyse memory access traces during model inference. It is made for research and helps study how model weights and cache are used.

## What it does

- runs model engines and captures memory access data
- parses raw trace data into a common format
- labels memory regions like weights and cache
- computes metrics such as retention and read frequency

## How to set it up

1. open a shell in the project folder
2. create and activate a Python virtual environment
3. install the project in editable mode with test dependencies

Example:

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[test]"
```

## How to run tests

Use pytest to run unit and integration tests.

```bash
pytest -m "unit or integration"
```

## Key files and folders

- `mrm_trace/` project source code
- `config/` yaml config files and schemas
- `tests/` unit and integration tests
- `CLAUDE.md` developer guide

## Notes

This project is built for Linux like systems with tools such as perf mem. It is mainly a research tool for studying memory use during model runs.
