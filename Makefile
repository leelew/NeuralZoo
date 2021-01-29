.PHONY: help init venv build source
.DEFAULT: help

# Makefile variables
VENV_NAME:=venv_lilu
PYTHON=${VENV_NAME}/bin/python3

help:
		@echo "make venv"
		@echo "[MetReg] auto-built development environment, use only once"
		@echo "make init"
		@echo "[MetReg] install required packages for MetReg, use only once"
		@echo "make install"
		@echo "[MetReg] install MetReg package"
		@echo "make source"
		@echo "[MetReg] source own development environment"

source:
		test -d $(VENV_NAME) || source $(VENV_NAME)/bin/activate

venv: $(VENV_NAME)/bin/activate 
$(VENV_NAME)/bin/activate: setup.py
		test -d $(VENV_NAME) || python3 -m venv $(VENV_NAME)
		${PYTHON} -m pip install -U pip
		${PYTHON} -m pip install -e .
		rm -f ./*.egg-info
		touch $(VENV_NAME)/bin/activate

init:
		pip3 install -r requirements.txt



