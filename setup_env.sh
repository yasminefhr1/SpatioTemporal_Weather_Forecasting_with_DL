#!/usr/bin/env bash
set -e

# Setup Python virtual env
PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_DIR=${VENV_DIR:-.venv}

echo ">>> Creating venv in ${VENV_DIR}"
${PYTHON_BIN} -m venv "${VENV_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo ">>> Upgrading pip"
python -m pip install --upgrade pip wheel setuptools

echo ">>> Installing requirements"
pip install -r requirements.txt

echo ">>> Done."
echo "Activate with: source ${VENV_DIR}/bin/activate"
