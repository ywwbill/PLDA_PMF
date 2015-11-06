/*
 * CorpusConvertor.cpp
 *
 *  Created on: 2015Äê11ÔÂ5ÈÕ
 *      Author: Yang Weiwei
 */

#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <memory.h>
using namespace std;

map<string, int> vocab2id;
vector<string> id2vocab;

void split(const string& src, const string& separator, vector<string>& dest)
{
    string str = src;
    string substring;
    string::size_type start = 0, index;
    dest.clear();

    do
    {
        index = str.find_first_of(separator,start);
        if (index != string::npos)
        {
            substring = str.substr(start,index-start);
            dest.push_back(substring);
            start = str.find_first_not_of(separator,index);
            if (start == string::npos) return;
        }
    }while(index != string::npos);

    //the last token
    substring = str.substr(start);
    dest.push_back(substring);
}


void readCorpus(char corpusFileName[])
{
	ifstream fin(corpusFileName);
	string line;
	vector<string> seg;
	while (getline(fin, line))
	{
		split(line, " ", seg);
		for (unsigned int i=0; i<seg.size(); i++)
		{
			if (seg[i].length()==0) continue;
			if (vocab2id.find(seg[i])==vocab2id.end())
			{
				vocab2id[seg[i]]=vocab2id.size()-1;
				id2vocab.push_back(seg[i]);
			}
		}
	}
	fin.close();
}

void readVocab(char vocabFileName[])
{
	ifstream fin(vocabFileName);
	string line;
	while (getline(fin, line))
	{
		vocab2id[line]=vocab2id.size()-1;
		id2vocab.push_back(line);
	}
	fin.close();
}

void writeVocab(char vocabFileName[])
{
	ofstream fout(vocabFileName);
	for (unsigned int i=0; i<id2vocab.size(); i++)
	{
		fout << id2vocab[i] << endl;
	}
	fout.close();
}

void convertCorpus(char srcCorpusFileName[], char destCorpusFileName[])
{
	ifstream fin(srcCorpusFileName);
	ofstream fout(destCorpusFileName);
	string line;
	vector<string> seg;
	int len,wordCount[id2vocab.size()];
	map<string, int>::iterator iter;
	while (getline(fin, line))
	{
		split(line, " ", seg);
		len=0;
		memset(wordCount, 0, sizeof(wordCount));
		for (unsigned int i=0; i<seg.size(); i++)
		{
			if (seg[i].length()==0) continue;
			iter=vocab2id.find(seg[i]);
			if (iter==vocab2id.end()) continue;
			wordCount[iter->second]++;
			len++;
		}

		fout << len;
		for (unsigned int i=0; i<id2vocab.size(); i++)
		{
			if (wordCount[i]>0)
			{
				fout << " " << i << ":" << wordCount[i];
			}
		}
		fout << endl;
	}
	fin.close();
	fout.close();
}

int main()
{
	char rawCorpusFileName[]="data/raw_corpus.txt";
	char rawTrainCorpusFileName[]="data/raw_train_corpus.txt";
	char rawTestCorpusFileName[]="data/raw_test_corpus.txt";

	char corpusFileName[]="data/corpus.txt";
	char trainCorpusFileName[]="data/train_corpus.txt";
	char testCorpusFileName[]="data/test_corpus.txt";

	char vocabFileName[]="data/vocab.txt";
//	readCorpus(rawCorpusFileName);
//	writeVocab(vocabFileName);
	readVocab(vocabFileName);
	convertCorpus(rawCorpusFileName, corpusFileName);
	convertCorpus(rawTrainCorpusFileName, trainCorpusFileName);
	convertCorpus(rawTestCorpusFileName, testCorpusFileName);
	return 0;
}
