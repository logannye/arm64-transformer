CLANG := xcrun clang
PYTHON ?= python3
WIDTH ?= 32
PRECISION ?= 0
BUILD := build/width-$(WIDTH)
FLAGS := -arch arm64 -O2 -I$(BUILD)
SME_FLAGS := -march=armv9-a+sme
OBJECTS := $(BUILD)/kernels.o $(BUILD)/linear.o $(BUILD)/linear_sme.o $(BUILD)/backend.o $(BUILD)/attention.o \
           $(BUILD)/softmax.o $(BUILD)/normalization.o $(BUILD)/precision.o $(BUILD)/model.o
.PHONY: all widths clean FORCE
all: $(BUILD)/transformer-arm64 $(BUILD)/libtransformer.dylib
widths:
	$(MAKE) WIDTH=32 all
	$(MAKE) WIDTH=64 all
	$(MAKE) WIDTH=128 all
	$(MAKE) WIDTH=256 all
	$(MAKE) WIDTH=512 all
	$(MAKE) WIDTH=1024 all
$(BUILD):
	mkdir -p $@
$(BUILD)/layout.inc: ../transformer/spec.py | $(BUILD)
	ARM64_TRANSFORMER_WIDTH=$(WIDTH) PYTHONPATH=.. $(PYTHON) -m transformer.spec --output $@
$(BUILD)/precision.h: FORCE | $(BUILD)
	@case "$(PRECISION)" in 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15) ;; *) exit 2 ;; esac
	@printf '#define TF_PRECISION_POLICY %s\n' '$(PRECISION)' > $@.tmp
	@cmp -s $@.tmp $@ && rm $@.tmp || mv $@.tmp $@
$(BUILD)/%.o: %.S $(BUILD)/layout.inc $(BUILD)/precision.h | $(BUILD)
	$(CLANG) $(FLAGS) -c $< -o $@
$(BUILD)/linear_sme.o: linear_sme.S $(BUILD)/layout.inc | $(BUILD)
	$(CLANG) $(FLAGS) $(SME_FLAGS) -c $< -o $@
$(BUILD)/backend.o: backend.S $(BUILD)/layout.inc | $(BUILD)
	$(CLANG) $(FLAGS) $(SME_FLAGS) -c $< -o $@
$(BUILD)/libtransformer.dylib: $(OBJECTS)
	$(CLANG) -arch arm64 -dynamiclib $(OBJECTS) -o $@
$(BUILD)/transformer-arm64: $(OBJECTS) $(BUILD)/runtime.o
	$(CLANG) -arch arm64 $(OBJECTS) $(BUILD)/runtime.o -o $@
clean:
	rm -f $(BUILD)/*.o $(BUILD)/libtransformer.dylib $(BUILD)/transformer-arm64
