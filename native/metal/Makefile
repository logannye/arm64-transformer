CLANG := xcrun clang++
BUILD := build
FLAGS := -std=c++17 -O2 -fobjc-arc -arch arm64 -mmacosx-version-min=15.0 -Wall -Wextra
SOURCE_HASH := $(shell shasum -a 256 backend.mm | cut -d' ' -f1)
.PHONY: all clean
all: $(BUILD)/libtransformer_metal.dylib
$(BUILD):
	mkdir -p $(BUILD)
$(BUILD)/libtransformer_metal.dylib: backend.mm Makefile | $(BUILD)
	$(CLANG) $(FLAGS) -DTFM_HOST_SHA256='"$(SOURCE_HASH)"' -dynamiclib backend.mm -framework Metal -framework Foundation -o $@
clean:
	rm -f $(BUILD)/libtransformer_metal.dylib
