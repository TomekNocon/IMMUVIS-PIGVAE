# Claude Code Instructions

This file contains instructions for Claude Code to help with development tasks in this project.

## Project Overview
This is the immuvis project - a Python-based project for immune system visualization and analysis.

## Development Commands

### Dependencies
- `uv sync` - Install/update dependencies
- `uv add <package>` - Add new dependency
- `uv remove <package>` - Remove dependency

### Code Quality
- `uv run ruff check` - Run linter
- `uv run ruff format` - Format code
- `uv run mypy src` - Type checking

### Testing
- `uv run pytest` - Run tests
- `uv run pytest tests/ -v` - Run tests with verbose output

### Environment
- Python 3.12.11 managed by uv
- Virtual environment automatically managed by uv

## Project Structure
- `src/` - Source code
- `tests/` - Test files
- `configs/` - Configuration files
- `notebooks/` - Jupyter notebooks
- `data/` - Data files
- `scripts/` - Utility scripts
