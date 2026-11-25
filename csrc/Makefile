CXX ?= g++
PORTABLE ?= 0
UNAME_M := $(shell uname -m)

ifeq ($(PORTABLE),1)
  ARCH_FLAGS ?=
else ifneq (,$(filter x86_64 amd64,$(UNAME_M)))
  HAS_AVX2 := $(shell grep -qw avx2 /proc/cpuinfo 2>/dev/null && grep -qw fma /proc/cpuinfo 2>/dev/null && echo 1 || echo 0)
  CAN_AVX2 := $(shell printf 'int main(){return 0;}\n' | $(CXX) -x c++ -mavx2 -mfma -march=native -c - -o /dev/null 2>/dev/null && echo 1 || echo 0)
  ifeq ($(HAS_AVX2)-$(CAN_AVX2),1-1)
    ARCH_FLAGS ?= -mavx2 -mfma -march=native
  else
    ARCH_FLAGS ?=
  endif
else
  ARCH_FLAGS ?=
endif

CAN_OMP := $(shell printf '#include <omp.h>\nint main(){return omp_get_max_threads();}\n' | $(CXX) -x c++ -fopenmp -c - -o /dev/null 2>/dev/null && echo 1 || echo 0)
ifeq ($(CAN_OMP),1)
  OMP_FLAGS ?= -fopenmp
else
  OMP_FLAGS ?=
endif

CXXFLAGS ?= -O3 $(ARCH_FLAGS) $(OMP_FLAGS) -Wall -Wextra -std=c++17
INCLUDES = -I.

TARGET = hybrid_infer

SRCS = main.cpp engine.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)

%.o: %.cpp engine.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
