PYTHON_FILES = $(wildcard python/*py)

.PHONY: help
help:
	@echo "\e[1m\e[93mHand Eye Calibration in MuJoCo\e[0m"
	@echo "make format-view"
	@echo "\e[90m- view the change done by black and format\e[0m"
	@echo "make format"
	@echo "\e[90m- format all python file used black and format\e[0m"
	@echo "make check"
	@echo "\e[90m- run mypy against python3 code\e[0m"

.PHONY: format-view
format-view:
	@python3 -m black $(PYTHON_FILES) -l 65535 --diff
	@python3 -m isort $(PYTHON_FILES) -l 65535 --diff

.PHONY: format
format:
	@python3 -m black $(PYTHON_FILES) -l 65535
	@python3 -m isort $(PYTHON_FILES) -l 65535

.PHONY: check
check:
	@python3 -m mypy --config-file mypy.ini

.PHONY: print
print:
	@echo $(PYTHON_FILES)
