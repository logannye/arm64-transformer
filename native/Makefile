CLANG := xcrun clang
BUILD := build
FLAGS := -arch arm64 -O2
OBJECTS := $(BUILD)/kernels.o $(BUILD)/linear.o $(BUILD)/attention.o \
           $(BUILD)/softmax.o $(BUILD)/normalization.o $(BUILD)/model.o
.PHONY: all clean
all: $(BUILD)/transformer-arm64 $(BUILD)/libtransformer.dylib
$(BUILD):
	mkdir -p $@
$(BUILD)/%.o: %.S layout.inc | $(BUILD)
	$(CLANG) $(FLAGS) -c $< -o $@
$(BUILD)/libtransformer.dylib: $(OBJECTS)
	$(CLANG) -arch arm64 -dynamiclib $(OBJECTS) -o $@
$(BUILD)/transformer-arm64: $(OBJECTS) $(BUILD)/runtime.o
	$(CLANG) -arch arm64 $(OBJECTS) $(BUILD)/runtime.o -o $@
clean:
	rm -f $(BUILD)/*.o $(BUILD)/libtransformer.dylib $(BUILD)/transformer-arm64
