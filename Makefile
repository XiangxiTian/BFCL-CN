# Makefile for Touch Event Processor

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O2
TARGET = touch_event_processor

# Source files
SOURCES = touch_event_processor.cpp
HEADERS = touch_event_processor.h

# Build target
$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

# Debug build
debug: CXXFLAGS = -std=c++11 -Wall -Wextra -g -DDEBUG
debug: $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)_debug

# Clean build artifacts
clean:
	rm -f $(TARGET) $(TARGET)_debug

# Run the program
run: $(TARGET)
	./$(TARGET)

# Help message
help:
	@echo "Available targets:"
	@echo "  make          - Build the release version"
	@echo "  make debug    - Build debug version with symbols"
	@echo "  make clean    - Remove built executables"
	@echo "  make run      - Build and run the program"
	@echo "  make help     - Show this help message"

.PHONY: clean run help debug
