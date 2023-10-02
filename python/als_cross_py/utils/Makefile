all: liblocalcross.so libutils.so

liblocalcross.so: localcross.o
	gcc -shared -o liblocalcross.so localcross.o

localcross.o: localcross.c
	gcc -c -fpic -O3 localcross.c -llapack -lblas

libutils.so: utils.o
	gcc -shared -o libutils.so utils.o

utils.o: utils.c
	gcc -c -fpic -O3 -Wall -pedantic -o utils.o utils.c -llapack -lblas -lpthread

clean:
	rm utils.o localcross.o