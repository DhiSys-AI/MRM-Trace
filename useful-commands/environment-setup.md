python3 -m venv venv
source venv/bin/activate

pip install pydantic pyyaml typer rich pandas pyarrow psutil numpy pytest pytest-cov pytest-mock freezegun hypothesis


# Verify key dependencies are installed
python -c 'import pydantic, yaml, typer, pandas, pytest;

# Run full test suite
python -m pytest tests/ -v --tb=short

python -m pytest tests/unit/config/test_validators.py::test_expand_sweep_correct_total -v --tb=long