#!/usr/bin/env bash
set -e

# Array of Python versions to build for
PYTHON_VERSIONS=("3.10" "3.11" "3.12")

# Clean dist directory
rm -rf dist/
mkdir -p dist/

# Build wheels for each Python version using Docker
for PY_VERSION in "${PYTHON_VERSIONS[@]}"; do
  echo "Building wheel for Python ${PY_VERSION}..."
  docker run --rm -v $(pwd):/io ghcr.io/pyo3/maturin build --release -i "python${PY_VERSION}" -o /io/dist
done

echo ""
echo "✅ Built wheels:"
ls -lh dist/

echo "run the follwing command to upload them to (test) PyPI:"
echo ""
echo "twine upload --repository testpypi ./dist/*"
echo ""
