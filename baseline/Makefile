.PHONY: venv install build check html reqif all clean

# NOTE: CI (.github/workflows/strictdoc.yml) pins Python 3.11 via
# actions/setup-python. This Makefile uses the "python3" on PATH so it
# works on hosts that don't have python3.11 installed; if your python3
# resolves to something older than 3.10, install 3.11 and point PYTHON
# at it explicitly, e.g. `make venv PYTHON=python3.11`.
PYTHON ?= python3

venv:
	$(PYTHON) -m venv .venv

install:
	.venv/bin/pip install -r requirements.txt

# "build" and "check" are the same operation: validate the whole
# requirements tree (grammar enforcement + composition) without
# producing HTML/ReqIF artifacts.
build: check

check:
	.venv/bin/strictdoc export . --formats=sdoc --output-dir _build

html:
	.venv/bin/strictdoc export . --formats=html --output-dir _build

reqif:
	.venv/bin/strictdoc export . --formats=reqif-sdoc --output-dir _build

all:
	.venv/bin/strictdoc export . --formats=html,reqif-sdoc,sdoc --output-dir _build

clean:
	rm -rf _build
