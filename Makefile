.PHONY: install test lint run-comps demo-comps

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
