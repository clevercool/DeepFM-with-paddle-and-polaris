CC       := g++
LIB_NAME :=
CFLAGS   := -fPIC -O0 -g -Wall -fpermissive -std=c++0x -mcmodel=large -fopenmp
LDFLAGS  := -L../polaris/lib -L./mkl/lib \
            -lpolaris \
            -lpthread -lrt -lgomp
INC      := -I./ \
            -I../polaris/include
OBJ_DIR  := ./obj
OUT_DIR  := ./lib
BIN_DIR  := ./bin

all: $(patsubst %.cpp, $(BIN_DIR)/%, $(wildcard *.cpp))

$(BIN_DIR)/%: $(OBJ_DIR)/%.o
	$(CC) $(INC) $(CFLAGS)  -o $@ $< $(LDFLAGS)

$(OBJ_DIR)/%.o: %.cpp dirmake
	$(CC) -c $(INC) $(CFLAGS) -o $@  $<

dirmake:
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(OBJ_DIR)

clean:
	rm -f $(OBJ_DIR)/*.o $(BIN_DIR)/* Makefile.bak

rebuild: clean build

