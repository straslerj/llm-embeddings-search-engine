MODE ?= default_mode

all: results/figures/$(MODE).png results/$(MODE)_results.csv

results/figures/$(MODE).png: responses/$(MODE)_responses.csv
	python3 evaluation.py --mode $(MODE)

responses/$(MODE)_responses.csv: create_responses.py
	python3 create_responses.py --mode $(MODE)
