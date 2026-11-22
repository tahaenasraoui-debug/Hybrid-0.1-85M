all: standalone python-ext

standalone:
	$(MAKE) -C csrc

python-ext:
	.venv/bin/python setup_cpp.py build_ext --inplace

test:
	PYTHONPATH=. .venv/bin/pytest tests/test_cpp_engine.py -v

bench:
	PYTHONPATH=. .venv/bin/python benchmarks/benchmark_systems.py

clean:
	$(MAKE) -C csrc clean
	rm -rf build *.so

.PHONY: all standalone python-ext test bench clean
