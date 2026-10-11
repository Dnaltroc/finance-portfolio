.PHONY: install test lint run-comps demo-comps app screenshots

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

demo-dashboard:
	python -m projects.dashboard build

# Streamlit app over the same engines (pip install -e ".[app]")
app:
	streamlit run app/Home.py

screenshots:
	python scripts/screenshots.py --url http://localhost:8599 --height 3200
