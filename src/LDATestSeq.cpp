/*
 * LDATest.cpp
 *
 *  Created on: 2015Äê11ÔÂ6ÈÕ
 *      Author: Yang Weiwei
 */

#include <stdlib.h>
#include <memory.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
using namespace std;

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

int main()
{
	char vocabFileName[]="data/vocab.txt";
	char testCorpusFileName[]="data/test_corpus.txt";
	char modelFileName[]="data/LDA_model.txt";

	double alpha=0.1;
	int numTopics=10,numDocs,numWords,numVocab,numIters=100;

	//read vocabulary
	vector<string> vocabulary;
	ifstream vocabFin(vocabFileName);
	string line;
	while (getline(vocabFin, line))
	{
		vocabulary.push_back(line);
	}
	vocabFin.close();
	numVocab=vocabulary.size();

	//read model
	ifstream modelFin(modelFileName);
	vector<string> seg;
	double phi[numTopics][numVocab];
	getline(modelFin, line);
	for (int topic=0; topic<numTopics; topic++)
	{
		getline(modelFin, line);
		split(line, " ", seg);
		for (int vocab=0; vocab<numVocab; vocab++)
		{
			phi[topic][vocab]=atof(seg[vocab].c_str());
		}
	}
	modelFin.close();

	//read corpus
	vector< vector<int> > corpus;
	ifstream corpusFin(testCorpusFileName);
	vector<string> segseg;
	int len,word,count;
	numWords=0;
	while (getline(corpusFin, line))
	{
		split(line, " ", seg);
		len=atoi(seg[0].c_str());
		numWords+=len;

		vector<int> doc;
		for (unsigned int i=1; i<seg.size(); i++)
		{
			split(seg[i], ":", segseg);
			word=atoi(segseg[0].c_str());
			count=atoi(segseg[1].c_str());
			for (int j=0; j<count; j++)
			{
				doc.push_back(word);
			}
		}
		corpus.push_back(doc);
	}
	corpusFin.close();
	numDocs=corpus.size();

	//init
	int docTopicCounts[numDocs][numTopics];
	memset(docTopicCounts, 0, sizeof(docTopicCounts));
	vector< vector<int> > topicAssigns;
	srand((unsigned)time(NULL));
	for (int doc=0; doc<numDocs; doc++)
	{
		vector<int> assign;
		for (unsigned int i=0; i<corpus[doc].size(); i++)
		{
			word=corpus[doc][i];
			int topic=rand()%numTopics;
			assign.push_back(topic);
			docTopicCounts[doc][topic]++;
		}
		topicAssigns.push_back(assign);
	}

	//sample
	int oldTopic,newTopic;
	double topicScores[numTopics],sum,sample;
	double logLikelihood,perplexity;
	double theta[numDocs][numTopics];
	for (int iter=1; iter<=numIters; iter++)
	{
		//sample each doc
		for (int doc=0; doc<numDocs; doc++)
		{
			//sample each topic assignment for each token
			for (unsigned int token=0; token<corpus[doc].size(); token++)
			{
				//remove old topic assignment
				word=corpus[doc][token];
				oldTopic=topicAssigns[doc][token];
				docTopicCounts[doc][oldTopic]--;

				//compute the probability of each topic
				sum=0.0;
				for (int topic=0; topic<numTopics; topic++)
				{
					topicScores[topic]=(docTopicCounts[doc][topic]+alpha)*phi[topic][word];
					sum+=topicScores[topic];
				}

				//sample new topic assignment
				sample=rand()/double(RAND_MAX)*sum;
				newTopic=-1;
				while (sample>0 && newTopic<numTopics-1)
				{
					newTopic++;
					sample-=topicScores[newTopic];
				}

				//add new topic assignment
				topicAssigns[doc][token]=newTopic;
				docTopicCounts[doc][newTopic]++;
			}
		}

		//compute theta
		for (int doc=0; doc<numDocs; doc++)
		{
			for (int topic=0; topic<numTopics; topic++)
			{
				theta[doc][topic]=(docTopicCounts[doc][topic]+alpha)/(corpus[doc].size()+numTopics*alpha);
			}
		}

		//compute log-likelihood and perplexity
		logLikelihood=0.0;
		for (int doc=0; doc<numDocs; doc++)
		{
			for (unsigned int token=0; token<corpus[doc].size(); token++)
			{
				word=corpus[doc][token];
				sum=0.0;
				for (int topic=0; topic<numTopics; topic++)
				{
					sum+=theta[doc][topic]*phi[topic][word];
				}
				logLikelihood+=log(sum);
			}
		}
		perplexity=exp(-logLikelihood/numWords);
		cout << "<" << iter << ">\tLog-LLD: " << logLikelihood << "\tPPX: " << perplexity << endl;
	}

	return 0;
}
