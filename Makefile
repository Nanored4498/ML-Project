EXEC=main

CC=gcc
CXX=g++
CXXFLAGS=-Wall -O3 -fopenmp
LDFLAGS=
INCLUDE=-I src/headers

RES_FOLDER=res
SRC_FOLDER=src
DEP_FOLDER=.d
BUILD_FOLDER=build
DEBUG_FOLDER=debug

SRC_C=$(wildcard $(SRC_FOLDER)/*.c)
DEP_C=$(patsubst $(SRC_FOLDER)/%.c, $(DEP_FOLDER)/%.d, $(SRC_C))
OBJ_C=$(patsubst $(SRC_FOLDER)/%.c, $(BUILD_FOLDER)/%.o, $(SRC_C))
SRC_CPP=$(wildcard $(SRC_FOLDER)/*.cpp)
OBJ_CPP=$(patsubst $(SRC_FOLDER)/%.cpp, $(BUILD_FOLDER)/%.opp, $(SRC_CPP))
DEP_CPP=$(patsubst $(SRC_FOLDER)/%.cpp, $(DEP_FOLDER)/%.d, $(SRC_CPP))

DEB=$(DEBUG_FOLDER)/$(EXEC)
DEB_C=$(patsubst $(SRC_FOLDER)/%.c, $(DEBUG_FOLDER)/%.o, $(SRC_C))
DEB_CPP=$(patsubst $(SRC_FOLDER)/%.cpp, $(DEBUG_FOLDER)/%.opp, $(SRC_CPP))

all: $(BUILD_FOLDER) $(EXEC) $(RES_FOLDER)

$(DEP_FOLDER) $(BUILD_FOLDER) $(RES_FOLDER):
	mkdir $@

$(EXEC): $(OBJ_C) $(OBJ_CPP)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILD_FOLDER)/%.o: $(SRC_FOLDER)/%.c
	$(CC) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

$(BUILD_FOLDER)/%.opp: $(SRC_FOLDER)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@


#DEBUG PART

.PHONY: deb
deb: $(DEBUG_FOLDER) $(DEB)
	valgrind ./$(DEB) ${ARGS}

$(DEBUG_FOLDER):
	mkdir $(DEBUG_FOLDER)

$(DEB): $(DEB_C) $(DEB_CPP)
	$(CXX) $(CXXFLAGS) -g $^ -o $@ $(LDFLAGS)

$(DEBUG_FOLDER)/%.o: $(SRC_FOLDER)/%.c
	$(CC) $(CXXFLAGS) $(INCLUDE) -g -c $< -o $@

$(DEBUG_FOLDER)/%.opp: $(SRC_FOLDER)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -g -c $< -o $@


#CLEAN PART

.PHONY: clean mrproper
clean:
	$(RM) -r $(DEP_FOLDER) $(BUILD_FOLDER) $(DEBUG_FOLDER)
mrproper: clean
	$(RM) $(EXEC)

#DEPENDENCIES

$(DEP_FOLDER)/%.d: $(SRC_FOLDER)/%.cpp $(DEP_FOLDER)
	$(CXX) $(INCLUDE) -MM -MD -o $@ $<

include $(DEP_C) $(DEP_CPP)