CC = gcc
CFLAGS = -Wall -Wextra -O3 -mavx2 -mfma -march=native -fopenmp
LDFLAGS = -lm -fopenmp

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

SOURCES = $(wildcard $(SRC_DIR)/*.c)
OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SOURCES))

EXAMPLES = $(wildcard examples/*.c)
EXAMPLE_BINARIES = $(patsubst examples/%.c,$(BIN_DIR)/%,$(EXAMPLES))

.PHONY: all clean examples

all: $(BIN_DIR)/libknn.a

$(BIN_DIR)/libknn.a: $(OBJECTS)
	@mkdir -p $(BIN_DIR)
	ar rcs $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

examples: $(BIN_DIR)/libknn.a $(EXAMPLE_BINARIES)

$(BIN_DIR)/%: examples/%.c $(BIN_DIR)/libknn.a
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $< -o $@ $(BIN_DIR)/libknn.a $(LDFLAGS)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

benchmark: examples
	@echo "Running benchmark..."
	@$(BIN_DIR)/iris_example

test: examples
	@echo "Running tests..."
	# Add test commands here

.PHONY: benchmark test

TESTS = $(wildcard tests/*.c)
TEST_BINARIES = $(patsubst tests/%.c,$(BIN_DIR)/%,$(TESTS))

# Add to the examples target
examples: $(BIN_DIR)/libknn.a $(EXAMPLE_BINARIES) $(TEST_BINARIES)

# Add test compilation rule
$(BIN_DIR)/%: tests/%.c $(BIN_DIR)/libknn.a
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $< -o $@ $(BIN_DIR)/libknn.a $(LDFLAGS)

# Add test target
test: $(TEST_BINARIES)
	@for test in $(TEST_BINARIES); do \
		echo "Running $$(basename $$test)..."; \
		./$$test; \
	done