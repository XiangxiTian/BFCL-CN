# Makefile for Touch Event Processor

CXX = g++
CXXFLAGS = -std=c++11 -Wall -O2
TARGET = touch_event_processor
SOURCES = touch_event_processor.cpp
HEADERS = touch_event_processor.h

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run