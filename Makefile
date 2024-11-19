init:
	( \
		python3 -m venv .venv; \
		source .venv/bin/activate; \
		python3 -m pip install --upgrade pip; \
		python3 -m pip install poetry; \
		poetry install; \
		pre-commit install; \
	)

validate: .pre-commit-config.yaml
	( \
		source .venv/bin/activate; \
		pre-commit run --all-files; \
	)

update:
	( \
		source .venv/bin/activate; \
		poetry self update; \
		poetry update; \
		pre-commit autoupdate; \
	)

clean:
	rm -rf .venv
	rm -rf .mypy_cache
	find . -name "*.pyc" -type f -delete
