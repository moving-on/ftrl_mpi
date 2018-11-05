
CC = mpicxx
CPPFLAGS = -Wall -O3 -fPIC -std=c++11 -march=native
INCLUDES = -I.
LDFLAGS = -pthread

all: ftrl_train

#.cpp.o:
#	$(CC) -c $^ $(INCLUDES) $(CPPFLAGS)
ftrl_train: ftrl_train.cpp *.h
	$(CC) -o $@ $^ $(INCLUDES) $(CPPFLAGS) $(LDFLAGS)

clean:
	rm -f *.o ftrl_train
