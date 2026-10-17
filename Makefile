.PHONY: install test lint run-comps demo-comps export-powerbi app screenshots demo-filing-tone train-filing-tone

install:
	pip install -e ".[dev]"

test:
	FINLIB_OFFLINE=1 pytest -q

lint:
	ruff check .

# Live run (needs .env with SEC_IDENTITY and FINNHUB_KEY)
run-comps:
	python -m projects.comps run --tickers AAPL,MSFT,GOOGL --peerset megacap

# Offline demo on fixtures, no keys needed
demo-comps:
	FINLIB_OFFLINE=1 python -m projects.comps run --tickers AAPL,MSFT,GOOGL --peerset demo

demo-variance:
	python -m projects.variance demo --period 2026-09

demo-lbo:
	python -m projects.lbo run --deal projects/lbo/deal.yaml && python -m projects.lbo summary

demo-sourcing:
	FINLIB_OFFLINE=1 python -m projects.sourcing run --day 2026-10-09

export-powerbi:
	python -m projects.powerbi export

demo-dashboard:
	python -m projects.dashboard build

demo-live:
	FINLIB_OFFLINE=1 python -m projects.pitch live

# Filing Tone (pip install -e ".[ml]"). Demo trains the CNN on the synthetic fixture, offline.
demo-filing-tone:
	FINLIB_OFFLINE=1 python -m projects.filing_tone train --model cnn --fixture --epochs 3 --device cpu

# Full run on Financial PhraseBank: baseline, CNN, DistilBERT, then the results table into the README
train-filing-tone:
	python -m projects.filing_tone train --model baseline && python -m projects.filing_tone train --model cnn && python -m projects.filing_tone train --model distilbert && python -m projects.filing_tone results

# Streamlit app over the same engines (pip install -e ".[app]")
app:
	streamlit run app/Home.py

screenshots:
	python scripts/screenshots.py --url http://localhost:8599 --height 3200
