.PHONY: data baselines main ablations theory figures report app test all

all: data baselines main ablations theory figures report

data:
	python src/main.py data

baselines:
	python src/main.py baselines

main:
	python src/main.py main

ablations:
	python src/main.py ablations

theory:
	python src/main.py theory

figures:
	python src/main.py figures

report:
	python src/main.py report

app:
	python src/app.py

test:
	pytest tests/
