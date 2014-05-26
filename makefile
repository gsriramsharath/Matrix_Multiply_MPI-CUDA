# Compiler Mode
DEBUG=no

# Compiler
CC=mpicc
CFLAGS=-O3
LDFLAGS=-lcudart -lcublas

EXEC=MatProd

SRC=perf.c matblock.c main.c 
OBJ=$(SRC:.c=.o)

ifeq ($(DEBUG),yes)
	CFLAGS=-Wall -g -DDEBUG
endif

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)
	@echo "Compilation Finished"

%.o: %.c
	$(CC) $(CFLAGS) -o $@ -c $<

.PHONY: clean

clean:
	@rm -rf *.o $(EXEC) 
	@echo "All objects and binary files deleted"
