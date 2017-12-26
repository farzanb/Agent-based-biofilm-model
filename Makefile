all:
	#g++ -std=c++11 -L/usr/local/lib -I/usr/local/include -Imat2 mat2/main2.c mat2/las2.c mat2/svdutil.c mat2/svdlib.c -lgsl -lgslcblas fibrin.cpp -o simulate
	g++ -L/usr/local/lib -I/usr/local/include -lgsl -lgslcblas fibrin.cpp -std=c++0x -o simulate
	#g++ -g -std=c++0x -I/opt/local/include/ -g -lm -Imat2 fibrin.cpp mat2/main2.c mat2/las2.c mat2/svdutil.c mat2/svdlib.c -o simulate -w
	
clean:
	rm simulate
