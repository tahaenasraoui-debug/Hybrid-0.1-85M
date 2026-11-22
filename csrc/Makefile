CXX ?= g++
CXXFLAGS = -O3 -mavx2 -mfma -fopenmp -march=native -Wall -Wextra -std=c++17
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
