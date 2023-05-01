ENVNAME = smear-beta
PYTHONVER = 3.9.7

install:
	@echo "Installing..."
	conda create -n $(ENVNAME) python==$(PYTHONVER) -y
	call conda activate $(ENVNAME) && pip install -r requirements.txt