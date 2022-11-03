DEMO_NAME     ?= aylien-datascience-demos
CONTAINER     ?= `echo $(DEMO_NAME) | tr "_" "-"`-demo
VERSION       ?= `cat VERSION`
PORT		  ?= 8000


.PHONY: dev
dev:
	pip install -r requirements.txt


.PHONY: run
run:
	streamlit \
		run Home.py \
		--logger.level debug \
		--server.port $(PORT) \
		--server.baseUrlPath $(DEMO_NAME)


.PHONY: build
build:
	cd ../..; docker build --no-cache -t $(CONTAINER):$(VERSION) -f demos/$(DEMO_NAME)/Dockerfile .
	@echo "To run the container, use:"
	@echo "docker run -p 8000:8000 -e --rm -it $(CONTAINER):$(VERSION)"
