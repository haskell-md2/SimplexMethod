make_dir:
	mkdir build

clear:
	rm -r build

all:
	clear
	make_dir
	g++ src/main.cpp -o build/main