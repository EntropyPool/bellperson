NVCC=nvcc
CLANG=clang++
# Target rules
all: build

build: bellperson





cpu: cpu_test.cpp
	$(EXEC) $(CLANG) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

gpu_test.o: gpu_test.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

gpu_test.so: gpu_test.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -shared -o $@ -c $<

gpu: gpu_test.cu
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	$(EXEC) ./bellperson

clean:
	rm -f fft.o multiexp.o cpu_test.o cpu gpu gpu_test.o gpu_test.so

clobber: clean
