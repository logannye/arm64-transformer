CLANG := xcrun clang
PYTHON ?= python3
WIDTH ?= 32
BUILD := build/width-$(WIDTH)
FLAGS := -arch arm64 -O2 -I$(BUILD)
OBJECTS := $(BUILD)/kernels.o $(BUILD)/linear.o $(BUILD)/attention.o \
           $(BUILD)/softmax.o $(BUILD)/normalization.o $(BUILD)/model.o
.PHONY: all widths clean
all: $(BUILD)/transformer-arm64 $(BUILD)/libtransformer.dylib
widths:
	$(MAKE) WIDTH=32 all
	$(MAKE) WIDTH=64 all
	$(MAKE) WIDTH=128 all
	$(MAKE) WIDTH=256 all
$(BUILD):
	mkdir -p $@
$(BUILD)/layout.inc: ../transformer/spec.py | $(BUILD)
	ARM64_TRANSFORMER_WIDTH=$(WIDTH) PYTHONPATH=.. $(PYTHON) -m transformer.spec --output $@
$(BUILD)/%.o: %.S $(BUILD)/layout.inc | $(BUILD)
	$(CLANG) $(FLAGS) -c $< -o $@
$(BUILD)/libtransformer.dylib: $(OBJECTS)
	$(CLANG) -arch arm64 -dynamiclib $(OBJECTS) -o $@
$(BUILD)/transformer-arm64: $(OBJECTS) $(BUILD)/runtime.o
	$(CLANG) -arch arm64 $(OBJECTS) $(BUILD)/runtime.o -o $@
clean:
	rm -f $(BUILD)/*.o $(BUILD)/libtransformer.dylib $(BUILD)/transformer-arm64
