SRC_DIR := src
OUT_DIR := out
MAIN    := handwrittendigitclassifier.Main
SOURCES := $(wildcard $(SRC_DIR)/handwrittendigitclassifier/*.java)
CLASSES := $(OUT_DIR)/handwrittendigitclassifier/Main.class

# Pin the recipe shell so `make clean` behaves the same from PowerShell, cmd, or Git Bash
ifeq ($(OS),Windows_NT)
SHELL   := cmd.exe
RMDIR   := if exist $(OUT_DIR) rmdir /s /q $(OUT_DIR)
else
RMDIR   := rm -rf $(OUT_DIR)
endif

.PHONY: run build clean

# Run from the project root so the relative data/ paths in Main resolve
run: build
	java -cp $(OUT_DIR) $(MAIN)

build: $(CLASSES)

# Recompile everything whenever any source file changes
$(CLASSES): $(SOURCES)
	javac -encoding UTF-8 -d $(OUT_DIR) $(SOURCES)

clean:
	$(RMDIR)
