.PHONY: help check-uv venv install install-dev install-all clean test test-unit test-notebook

# Default Python version
PYTHON := python3
VENV := venv
BIN := $(VENV)/bin
PYTHON_VENV := $(BIN)/python
UV := uv
NOTEBOOK := examples/SOPRA_Demo.ipynb

help:
	@echo "Available targets:"
	@echo "  make check-uv       - Check if uv is installed (and offer to install if not)"
	@echo "  make venv           - Create virtual environment using uv"
	@echo "  make install        - Create venv and install package with core dependencies"
	@echo "  make install-dev    - Create venv and install package with dev dependencies"
	@echo "  make install-all    - Create venv and install package with all dependencies"
	@echo "  make test           - Execute Jupyter notebook with current venv"
	@echo "  make test-unit      - Run pytest unit tests (requires dev dependencies)"
	@echo "  make test-notebook  - Execute Jupyter notebook (same as test)"
	@echo "  make clean          - Remove virtual environment and build artifacts"

check-uv:
	@if ! which $(UV) > /dev/null 2>&1; then \
		echo "❌ uv is not installed."; \
		echo ""; \
		read -p "Would you like to install uv now? (y/n): " answer; \
		if [ "$$answer" = "y" ] || [ "$$answer" = "Y" ]; then \
			echo ""; \
			echo "📥 Installing uv..."; \
			curl -LsSf https://astral.sh/uv/install.sh | sh; \
			echo ""; \
			echo "✅ uv has been installed!"; \
			echo "⚠️  Please restart your shell or run: source $$HOME/.cargo/env"; \
			echo "   Then run 'make install-all' again."; \
			exit 1; \
		else \
			echo ""; \
			echo "❌ Installation cancelled. uv is required to continue."; \
			echo "   You can install it manually later with:"; \
			echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"; \
			exit 1; \
		fi; \
	else \
		echo "✅ uv is installed (version: $$($(UV) --version))"; \
	fi

venv: check-uv
	@echo "Creating virtual environment with uv..."
	$(UV) venv $(VENV)
	@echo "Virtual environment created at ./$(VENV)"

install: venv
	@echo "Installing package with core dependencies using uv..."
	$(UV) pip install --python $(PYTHON_VENV) -e .
	@echo ""
	@echo "✅ Installation complete!"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "To activate the virtual environment, run:"
	@echo "  source $(BIN)/activate"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

install-dev: venv
	@echo "Installing package with development dependencies using uv..."
	$(UV) pip install --python $(PYTHON_VENV) -e ".[dev]"
	@echo ""
	@echo "✅ Installation complete!"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "To activate the virtual environment, run:"
	@echo "  source $(BIN)/activate"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

install-all: venv
	@echo "Installing package with all dependencies using uv..."
	$(UV) pip install --python $(PYTHON_VENV) -e ".[all]"
	@echo ""
	@echo "✅ Installation complete!"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "To activate the virtual environment, run:"
	@echo "  source $(BIN)/activate"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

test: test-notebook

test-notebook:
	@if [ ! -d "$(VENV)" ]; then \
		echo "❌ Virtual environment not found. Run 'make install-all' first."; \
		exit 1; \
	fi
	@if [ ! -f "$(NOTEBOOK)" ]; then \
		echo "❌ Notebook '$(NOTEBOOK)' not found."; \
		exit 1; \
	fi
	@echo "📓 Executing Jupyter notebook: $(NOTEBOOK)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	$(BIN)/jupyter nbconvert --to notebook --execute $(NOTEBOOK) \
		--ExecutePreprocessor.timeout=600 \
		--output SOPRA_Demo.ipynb
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "✅ Notebook executed successfully!"

test-unit:
	@if [ ! -d "$(VENV)" ]; then \
		echo "❌ Virtual environment not found. Run 'make install-dev' first."; \
		exit 1; \
	fi
	@echo "🧪 Running pytest unit tests..."
	$(BIN)/pytest

clean:
	@echo "Cleaning up..."
	@if [ "$$VIRTUAL_ENV" != "" ] && [ "$$(basename $$VIRTUAL_ENV)" = "$(VENV)" ]; then \
		echo "⚠️  Virtual environment is currently activated. Please deactivate it first with:"; \
		echo "   deactivate"; \
		echo "Then run 'make clean' again."; \
		exit 1; \
	fi
	rm -rf $(VENV)
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type d -name '.pytest_cache' -exec rm -rf {} +
	find . -type d -name '.coverage' -exec rm -rf {} +
	find . -type f -name '.coverage' -delete
	find . -type d -name 'htmlcov' -exec rm -rf {} +
	@echo "Cleanup complete!"
