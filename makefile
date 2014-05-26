# Compiler Mode
DEBUG=no

# Compiler
CC=mpicxx

CFLAGS=-O3
LDFLAGS=-lcudart -lcublas

NVCC=nvcc
NVCC_FLAGS= -O3 -arch=sm_20

EXEC=MatProd

SRC=perf.c worker.c main.c
SRCGPU=matblock.cu
OBJ=$(SRC:.c=.o) $(SRCGPU:.cu=.o)

ifeq ($(DEBUG),yes)
	CFLAGS=-Wall -g -DDEBUG
	NVCC_FLAGS=-g -DDEBUG
endif

all: $(EXEC)

MatProd: $(OBJ)
	@echo "Compiling..."
	$(CC) -o $@ $^ $(LDFLAGS)
	@echo "Finished"

%.o: %.c
	@$(CC) $(CFLAGS) -o $@ -c $<

%.o: %.cu
	@$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

clean:
	@rm -rf *.o $(EXEC) 
		@echo "All objects and binary files deleted"
