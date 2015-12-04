all: bin/CorpusConvertor.exe bin/LDATrainSeq.exe bin/LDATestSeq.exe bin/ResultComparer.exe

bin/CorpusConvertor.exe: obj/CorpusConvertor.o
	g++ obj/CorpusConvertor.o -o bin/CorpusConvertor.exe

obj/CorpusConvertor.o: src/CorpusConvertor.cpp
	g++ -O0 -g3 -Wall -c src/CorpusConvertor.cpp -o obj/CorpusConvertor.o
	
bin/LDATrainSeq.exe: obj/LDATrainSeq.o
	g++ obj/LDATrainSeq.o -o bin/LDATrainSeq.exe

obj/LDATrainSeq.o: src/LDATrainSeq.cpp
	g++ -O0 -g3 -Wall -c src/LDATrainSeq.cpp -o obj/LDATrainSeq.o
	
bin/LDATestSeq.exe: obj/LDATestSeq.o
	g++ obj/LDATestSeq.o -o bin/LDATestSeq.exe

obj/LDATestSeq.o: src/LDATestSeq.cpp
	g++ -O0 -g3 -Wall -c src/LDATestSeq.cpp -o obj/LDATestSeq.o
	
bin/ResultComparer.exe: obj/ResultComparer.o
	g++ obj/ResultComparer.o -o bin/ResultComparer.exe

obj/ResultComparer.o: src/ResultComparer.cpp
	g++ -O0 -g3 -Wall -c src/ResultComparer.cpp -o obj/ResultComparer.o