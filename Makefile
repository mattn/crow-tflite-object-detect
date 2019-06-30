SRCS = \
	main.cxx

OBJS = $(subst .cc,.o,$(subst .cxx,.o,$(subst .cpp,.o,$(SRCS))))

TENSORFLOW_ROOT = $(shell go env GOPATH)/src/github.com/tensorflow/tensorflow

CXXFLAGS = -std=c++17 $(shell pkg-config --cflags opencv4)
LDFLAGS = -std=c++17
LIBS = -lboost_system -lpthread -lws2_32 -lmswsock $(shell pkg-config --libs opencv4)
TARGET = crow-tflite-object-detect
ifeq ($(OS),Windows_NT)
TARGET := $(TARGET).exe
TARGET_ARCH = windows_x86_64
else
TARGET_ARCH = linux_x86_64
endif

CXXFLAGS += -DMG_ENABLE_HTTP_STREAMING_MULTIPART=1 \
	-I$(TENSORFLOW_ROOT) \
	-I$(TENSORFLOW_ROOT)/tensorflow/lite/tools/make/downloads/flatbuffers/include

LIBS += -L$(TENSORFLOW_ROOT)/tensorflow/lite/tools/make/gen/$(TARGET_ARCH)/lib \
	-ltensorflow-lite \
	-lstdc++ \
	-lpthread \
	-ldl \
	-lm

.SUFFIXES: .cpp .cxx .o

all : $(TARGET)

$(TARGET) : $(OBJS)
	g++ $(LDFLAGS) -o $@ $(OBJS) $(LIBS)

.cxx.o :
	g++ -c $(CXXFLAGS) -I. $< -o $@

.cpp.o :
	g++ -c $(CXXFLAGS) -I. $< -o $@

clean :
	rm -f *.o $(TARGET)
