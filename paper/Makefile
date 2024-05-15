DOCKER_CMD = docker run --rm -it -v $(PWD):/data -u $$(id -u):$$(id -g)
PLATFORM = linux/amd64

.PHONY: build

build:
	$(DOCKER_CMD) openjournals/inara -o pdf,crossref paper.md
