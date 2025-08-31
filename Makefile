CXX := g++
CXXFLAGS := -Wall -O2

PORT ?= /dev/ttyS3
TEST_ID ?= 6

MAIN_TARGET := main
MAIN_SRC := main.cpp

$(MAIN_TARGET): $(MAIN_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^

run: $(MAIN_TARGET)
	./$(MAIN_TARGET) $(PORT)

test:
	$(CXX) $(CXXFLAGS) -o tests/test tests/test.cpp
	./tests/test $(PORT) $(TEST_ID)

clean:
	rm -f $(MAIN_TARGET) tests/test
