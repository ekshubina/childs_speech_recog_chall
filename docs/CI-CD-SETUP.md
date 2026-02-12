# CI/CD and Development Tools Setup

## Overview
Automated testing, linting, and code quality checks have been configured for the project using GitHub Actions and standard Python development tools.

## Files Created

### 1. GitHub Actions Workflow (`.github/workflows/test.yml`)
- **Purpose**: Automated CI/CD pipeline for testing and code quality
- **Triggers**: Push and pull requests to `main` and `develop` branches
- **Jobs**:
  - **test**: Runs full test suite across Python 3.8-3.11
    - Linting with flake8
    - Code formatting checks with black
    - Import sorting checks with isort  
    - Unit/integration tests with pytest
    - Coverage reporting to Codecov
  - **lint-only**: Fast linting checks without heavy dependencies
    - Includes optional mypy type checking

### 2. Pytest Configuration (`pytest.ini`)
- Test discovery: Finds all `test_*.py` files in `tests/` directory
- Custom markers for organizing tests:
  - `@pytest.mark.unit` - Unit tests
  - `@pytest.mark.integration` - Integration tests
  - `@pytest.mark.slow` - Long-running tests
  - `@pytest.mark.requires_gpu` - GPU-dependent tests
  - `@pytest.mark.requires_audio` - Audio file-dependent tests
- Coverage configuration with exclusions for boilerplate code

### 3. Flake8 Configuration (`.flake8`)
- Max line length: 127 characters
- Max complexity: 10
- Ignores: E203, W503 (black compatibility)
- Excludes: Generated files, virtual environments, data directories

### 4. Black & Isort Configuration (`pyproject.toml`)
- **Black**: Code formatter with 127 character line length
- **Isort**: Import sorter with black-compatible profile
- **MyPy**: Optional type checking configuration

### 5. Development Dependencies (`requirements-dev.txt`)
- Testing: pytest, pytest-cov, pytest-xdist, pytest-timeout
- Formatting: black, isort
- Linting: flake8, mypy

## Usage

### Running Tests Locally
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific markers
pytest -m unit
pytest -m integration
pytest -m "not slow"
```

### Code Formatting
```bash
# Format code
black src/ scripts/ tests/

# Sort imports
isort src/ scripts/ tests/

# Check without modifying
black --check src/ scripts/ tests/
isort --check-only src/ scripts/ tests/
```

### Linting
```bash
# Run flake8
flake8 src/ scripts/ tests/

# Run mypy type checking
mypy src/ --ignore-missing-imports
```

### Pre-commit Checks (Recommended)
```bash
# Run all checks before committing
black src/ scripts/ tests/ && \
isort src/ scripts/ tests/ && \
flake8 src/ scripts/ tests/ && \
pytest tests/ -v
```

## CI/CD Pipeline

### Automatic Checks on Pull Requests
1. Code syntax validation (flake8 critical errors)
2. Code formatting verification (black, isort)
3. Full test suite execution across Python versions
4. Coverage report generation

### Required Passing Checks
- All tests must pass
- No flake8 critical errors (E9, F63, F7, F82)
- Code must be formatted with black
- Imports must be sorted with isort

### Optional Checks
- MyPy type checking (continues on error)
- Non-critical flake8 warnings
- Coverage thresholds (informational only)

## Benefits
- **Consistency**: Automated code formatting ensures uniform style
- **Quality**: Early detection of bugs and code smells
- **Confidence**: Test suite prevents regressions
- **Collaboration**: Pull requests automatically validated before merge
- **Documentation**: Test markers help organize and filter test execution
