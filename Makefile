init:
	pip install -r requirements.txt

docs: docs/source alias
	make -Cdocs html
