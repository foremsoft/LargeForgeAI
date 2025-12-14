# Contributing to LargeForgeAI

Thank you for your interest in contributing to LargeForgeAI! This guide will help you get started.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Making Changes](#making-changes)
5. [Pull Request Process](#pull-request-process)
6. [Coding Standards](#coding-standards)
7. [Testing Guidelines](#testing-guidelines)
8. [Documentation](#documentation)

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please read and follow our [Code of Conduct](./CODE_OF_CONDUCT.md).

**Key Points:**
- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- No harassment or discrimination

---

## Getting Started

### Types of Contributions

| Contribution | Description | Label |
|--------------|-------------|-------|
| Bug Fix | Fix an existing issue | `bug` |
| Feature | Add new functionality | `enhancement` |
| Documentation | Improve docs | `documentation` |
| Tests | Add or improve tests | `testing` |
| Performance | Optimize code | `performance` |

### Finding Issues

1. Check [GitHub Issues](https://github.com/largeforgeai/largeforgeai/issues)
2. Look for `good first issue` label for beginners
3. Check `help wanted` for priority items
4. Comment on an issue before starting work

### Reporting Bugs

Create an issue with:
- Clear title and description
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, GPU)
- Error messages and logs

### Suggesting Features

Create an issue with:
- Use case description
- Proposed solution
- Alternatives considered
- Willingness to implement

---

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- NVIDIA GPU (optional for most development)

### Setup Steps

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/largeforgeai.git
cd largeforgeai

# 3. Add upstream remote
git remote add upstream https://github.com/largeforgeai/largeforgeai.git

# 4. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows

# 5. Install development dependencies
pip install -e ".[dev]"

# 6. Install pre-commit hooks
pre-commit install

# 7. Verify setup
pytest tests/unit/ -v --co  # List tests
largeforge --version
```

### IDE Setup

**VS Code:**
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.formatting.provider": "none",
    "editor.formatOnSave": true,
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.codeActionsOnSave": {
            "source.fixAll": "explicit",
            "source.organizeImports": "explicit"
        }
    },
    "python.linting.enabled": true,
    "python.linting.mypyEnabled": true
}
```

**PyCharm:**
- Set Python interpreter to `.venv/bin/python`
- Enable Ruff plugin
- Configure mypy integration

---

## Making Changes

### Branching Strategy

```
main
  │
  ├── feature/add-streaming-support
  │
  ├── fix/memory-leak-inference
  │
  └── docs/api-reference-update
```

**Branch naming:**
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation
- `refactor/description` - Code refactoring
- `test/description` - Test additions

### Development Workflow

```bash
# 1. Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# 2. Create feature branch
git checkout -b feature/my-feature

# 3. Make changes
# ... edit files ...

# 4. Run tests
pytest tests/unit/ -v

# 5. Run linting
ruff check .
ruff format .

# 6. Run type checking
mypy llm/

# 7. Commit changes
git add .
git commit -m "feat: add streaming support for chat completions"

# 8. Push to your fork
git push origin feature/my-feature

# 9. Create Pull Request on GitHub
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting (no code change)
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Examples:**
```
feat(inference): add streaming support for completions

fix(training): resolve memory leak in gradient accumulation

docs(api): update authentication section

refactor(router): simplify classifier interface
```

---

## Pull Request Process

### Before Submitting

- [ ] Tests pass locally
- [ ] Code is formatted (`ruff format .`)
- [ ] Linting passes (`ruff check .`)
- [ ] Type checking passes (`mypy llm/`)
- [ ] Documentation updated if needed
- [ ] Commit messages follow convention

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing performed.

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Changelog entry added
```

### Review Process

1. **Automated checks** - CI runs tests, linting, type checking
2. **Code review** - At least one maintainer approval
3. **Discussion** - Address feedback
4. **Merge** - Squash and merge when approved

### After Merging

- Delete your feature branch
- Sync your fork with upstream
- Close related issues

---

## Coding Standards

### Python Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting.

```toml
# pyproject.toml
[tool.ruff]
target-version = "py310"
line-length = 88

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
]

[tool.ruff.format]
quote-style = "double"
```

### Type Hints

All public functions must have type hints:

```python
# Good
def generate(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> GenerationResult:
    """Generate text completion."""
    ...

# Bad - no type hints
def generate(prompt, max_tokens=256, temperature=0.7):
    ...
```

### Docstrings

Use Google style docstrings:

```python
def train(
    model: str,
    dataset: Dataset,
    config: TrainingConfig,
) -> TrainingResult:
    """Train a model on the given dataset.

    Args:
        model: Model name or path.
        dataset: Training dataset.
        config: Training configuration.

    Returns:
        Training result containing metrics and model path.

    Raises:
        ValueError: If model is not found.
        RuntimeError: If training fails.

    Example:
        >>> result = train("mistral-7b", dataset, config)
        >>> print(result.final_loss)
        0.523
    """
```

### Error Handling

```python
# Use specific exceptions
class ModelNotFoundError(LargeForgeError):
    """Raised when model is not found."""
    pass

# Provide helpful error messages
def load_model(path: str) -> Model:
    if not Path(path).exists():
        raise ModelNotFoundError(
            f"Model not found at {path}. "
            f"Run 'largeforge model download {path}' to download."
        )
```

---

## Testing Guidelines

### Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_data.py
│   ├── test_training.py
│   ├── test_inference.py
│   └── test_router.py
├── integration/             # Integration tests
│   ├── test_api.py
│   └── test_workflows.py
├── conftest.py              # Shared fixtures
└── fixtures/                # Test data
```

### Writing Tests

```python
# tests/unit/test_training.py
import pytest
from llm.training import TrainingConfig, SFTTrainer

class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self):
        """Test config has sensible defaults."""
        config = TrainingConfig(
            model_name="test-model",
            output_dir="./output"
        )
        assert config.num_epochs == 3
        assert config.learning_rate == 2e-5

    def test_invalid_learning_rate(self):
        """Test validation rejects negative learning rate."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            TrainingConfig(
                model_name="test",
                output_dir="./output",
                learning_rate=-1.0
            )

    @pytest.mark.parametrize("lr", [1e-6, 1e-5, 1e-4])
    def test_valid_learning_rates(self, lr):
        """Test various valid learning rates."""
        config = TrainingConfig(
            model_name="test",
            output_dir="./output",
            learning_rate=lr
        )
        assert config.learning_rate == lr
```

### Test Fixtures

```python
# tests/conftest.py
import pytest
from pathlib import Path

@pytest.fixture
def sample_dataset():
    """Provide sample training dataset."""
    return [
        {"instruction": "Test", "input": "", "output": "Response"}
    ]

@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir

@pytest.fixture
def mock_model(mocker):
    """Provide mocked model for tests."""
    model = mocker.MagicMock()
    model.generate.return_value = "Generated text"
    return model
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=llm --cov-report=html

# Run specific test file
pytest tests/unit/test_training.py

# Run tests matching pattern
pytest -k "test_config"

# Run with verbose output
pytest -v

# Run only fast tests
pytest -m "not slow"
```

---

## Documentation

### Documentation Types

| Type | Location | Purpose |
|------|----------|---------|
| API Docs | `docs/api/` | Technical reference |
| Guides | `docs/guides/` | How-to tutorials |
| Concepts | `docs/concepts/` | Explanations |
| Examples | `examples/` | Code examples |

### Updating Documentation

1. Keep docs in sync with code
2. Use clear, concise language
3. Include examples
4. Test code examples

### Building Docs

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build documentation
mkdocs build

# Serve locally
mkdocs serve
# Visit http://localhost:8000
```

---

## Getting Help

- **Discord**: [discord.gg/largeforge](https://discord.gg/largeforge)
- **GitHub Discussions**: For questions and ideas
- **GitHub Issues**: For bugs and features

---

*Thank you for contributing to LargeForgeAI!*
