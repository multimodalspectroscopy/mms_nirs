version:
	@poetry version $(v)
	@git add pyproject.toml
	@git commit -m "v$$(poetry version -s)"
	@git tag v$$(poetry version -s)
	@git push
	@git push --tags
	@poetry version

publish-test:
	@poetry build
	@poetry publish -r test-pypi

publish-prod:
	@poetry build
	@poetry publish
