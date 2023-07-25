CC = nvcc
CFLAGS = -O3 -std=c++20

SRC = $(wildcard src/*.cu)
SRC += $(wildcard src/vector/*.cu)
OBJ = $(patsubst %.cu, %.o, ${SRC})

TARGET := build/RayTracing

${OBJ}:%.o:%.cu
	$(CC) -c $< -o $@ $(CFLAGS)

$(TARGET): $(OBJ)
	$(CC) -o $(TARGET) $^ $(CFLAGS)

.PHONY : ALL
ALL : $(TARGET)
	./$(TARGET)

.PHONY : cmALL
cmALL:
	cmake --build build --config Release --parallel $(nproc)
	./$(TARGET)

.PHONY : perf
perf:
	nvprof ./$(TARGET)

.PHONY : build
build: $(TARGET)

.PHONY : fmt
fmt:
	clang-format -i $(shell find src -name '*.cu')
	clang-format -i $(shell find src -name '*.cuh')

.PHONY : clean
clean:
	rm src/*.o
