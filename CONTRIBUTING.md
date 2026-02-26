# Contributing to Vision Agent

Thank you for considering contributing to Vision Agent! 🎉

## How to Contribute

### Reporting Bugs

1. Check existing [issues](https://github.com/rupac4530-creator/vision-agent/issues) first
2. Open a new issue with a clear title and description
3. Include steps to reproduce, expected vs actual behavior
4. Add labels: `bug`, `enhancement`, `documentation`, etc.

### Suggesting Features

1. Open an issue with the `enhancement` label
2. Describe the feature and its use case
3. If possible, include mockups or code snippets

### Pull Requests

1. **Fork** the repo and create your branch from `main`
2. **Install** dependencies: `cd backend && pip install -r requirements.txt`
3. **Make** your changes
4. **Test** your changes: `python test_deep_sdk.py`
5. **Commit** with a clear message: `git commit -m "feat: add new feature"`
6. **Push** and open a Pull Request

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new AI tab for drone surveillance
fix: resolve LLM cascade timeout issue
docs: update API reference
test: add tests for RAG engine
chore: update dependencies
refactor: simplify event bus architecture
```

### Code Style

- **Python**: Follow PEP 8. Use type hints where possible.
- **Docstrings**: Use Google-style docstrings for public functions.
- **Tests**: Add tests for new features. Run `python test_deep_sdk.py` to verify.

### Adding a New SDK Module

1. Create `backend/your_module.py` with clear docstrings
2. Add tests in `backend/test_deep_sdk.py`
3. Wire into `backend/main.py` (import + endpoint)
4. Update `backend/audit.json` with the module entry
5. Update this README's SDK Modules table

### Adding a New AI Tab

1. Add backend logic in a new `backend/your_tab.py`
2. Add endpoint(s) in `backend/main.py`
3. Add UI tab in `backend/static/index.html`
4. Update the Features table in README

### Environment Variables

- **Never** commit API keys or secrets
- Add new variables to `.env.example` with placeholder values
- Document in README's Environment Variables section

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/vision-agent.git
cd vision-agent/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_deep_sdk.py

# Start dev server
python -m uvicorn main:app --reload --port 8000
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
