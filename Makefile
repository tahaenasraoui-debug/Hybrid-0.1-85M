PYTHON ?= python3

all: standalone python-ext

standalone:
	$(MAKE) -C csrc

python-ext:
	$(PYTHON) setup_cpp.py build_ext --inplace

test:
	PYTHONPATH=. $(PYTHON) -m pytest tests/test_cpp_engine.py -v

bench:
	PYTHONPATH=. $(PYTHON) benchmarks/benchmark_systems.py

clean:
	$(MAKE) -C csrc clean
	rm -rf build *.so

.PHONY: all standalone python-ext test bench clean
