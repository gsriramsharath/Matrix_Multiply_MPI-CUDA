# Compiler Mode
DEBUG=no

# Compiler
CC=mpicxx
NVCC=nvcc
CFLAGS=-O3
LDFLAGS=-lm -lcudart -lcublas
EXEC=MatProdIO
SRC=perf.c worker.c main.c
SRCGPU=matblock.cu
OBJ=$(SRC:.c=.o) $(SRCGPU:.cu=.o)

ifeq ($(DEBUG),yes)
	CFLAGS=-Wall -O3 -DDEBUG
endif

all:clean $(EXEC)

MatProdIO: $(OBJ)
	@echo "Compiling..."
	$(CC) -o $@ $^ $(LDFLAGS)
	@echo "Finished"

%.o: %.c
	@$(CC) $(CFLAGS) -o $@ -c $<

%.o: %.cu
	@$(NVCC) $(CFLAGS) -o $@ -c $<

clean:
	@rm -rf *.o $(EXEC) 
		@echo "Old .o deleted"
