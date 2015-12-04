#include <stdlib.h>
#include <memory.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
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

int main(int argc, char** argv)
{
	int numTopics=10,numVocab=7167;

	char trainSeqThetaFileName[]="data/lda/train_seq_theta.txt";
	vector< vector<double> > trainSeqTheta;
	ifstream finTrainSeqTheta(trainSeqThetaFileName);
	string line;
	vector<string> seg;
	while (getline(finTrainSeqTheta, line))
	{
		split(line, " ", seg);
		vector<double> doc;
		for (int topic=0; topic<numTopics; topic++)
		{
			doc.push_back(atof(seg[topic].c_str()));
		}
		trainSeqTheta.push_back(doc);
	}
	finTrainSeqTheta.close();

	char testSeqThetaFileName[]="data/lda/test_seq_theta.txt";
	vector< vector<double> > testSeqTheta;
	ifstream finTestSeqTheta(testSeqThetaFileName);
	while (getline(finTestSeqTheta, line))
	{
		split(line, " ", seg);
		vector<double> doc;
		for (int topic=0; topic<numTopics; topic++)
		{
			doc.push_back(atof(seg[topic].c_str()));
		}
		testSeqTheta.push_back(doc);
	}
	finTestSeqTheta.close();

	char seqPhiFileName[]="data/lda/seq_phi.txt";
	double seqPhi[numTopics][numVocab];
	ifstream finSeqPhi(seqPhiFileName);
	for (int topic=0; topic<numTopics; topic++)
	{
		getline(finSeqPhi, line);
		split(line, " ", seg);
		for (int vocab=0; vocab<numVocab; vocab++)
		{
			seqPhi[topic][vocab]=atof(seg[vocab].c_str());
		}
	}
	finSeqPhi.close();

	char trainParThetaFileName[]="data/lda/train_par_theta.txt";
	vector< vector<double> > trainParTheta;
	ifstream finTrainParTheta(trainParThetaFileName);
	while (getline(finTrainParTheta, line))
	{
		split(line, " ", seg);
		vector<double> doc;
		for (int topic=0; topic<numTopics; topic++)
		{
			doc.push_back(atof(seg[topic].c_str()));
		}
		trainParTheta.push_back(doc);
	}
	finTrainParTheta.close();

	char testParThetaFileName[]="data/lda/test_par_theta.txt";
	vector< vector<double> > testParTheta;
	ifstream finTestParTheta(testParThetaFileName);
	while (getline(finTestParTheta, line))
	{
		split(line, " ", seg);
		vector<double> doc;
		for (int topic=0; topic<numTopics; topic++)
		{
			doc.push_back(atof(seg[topic].c_str()));
		}
		testParTheta.push_back(doc);
	}
	finTestParTheta.close();

	char parPhiFileName[]="data/lda/par_phi.txt";
	double parPhi[numTopics][numVocab];
	ifstream finParPhi(parPhiFileName);
	for (int topic=0; topic<numTopics; topic++)
	{
		getline(finParPhi, line);
		split(line, " ", seg);
		for (int vocab=0; vocab<numVocab; vocab++)
		{
			parPhi[topic][vocab]=atof(seg[vocab].c_str());
		}
	}
	finParPhi.close();

	int topicMatch[numTopics];
	bool matched[numTopics];
	memset(matched, false, sizeof(matched));
	for (int t1=0; t1<numTopics; t1++)
	{
		double minKLD=100000000;
		int minTopic=-1;
		for (int t2=0; t2<numTopics; t2++)
		{
			double kld=0.0;
			for (int vocab=0; vocab<numVocab; vocab++)
			{
				kld+=parPhi[t1][vocab]*log2(parPhi[t1][vocab]/seqPhi[t2][vocab]);
			}
			if (kld<minKLD)
			{
				minKLD=kld;
				minTopic=t2;
			}
		}

		if (matched[minTopic])
		{
			cout << "error" << endl;
		}
		matched[minTopic]=true;
		topicMatch[t1]=minTopic;
	}

	double avgKLD=0.0;
	for (int doc=0; doc<trainParTheta.size(); doc++)
	{
		double kld=0.0;
		for (int topic=0; topic<numTopics; topic++)
		{
			kld+=trainParTheta[doc][topic]*log2(trainParTheta[doc][topic]/trainSeqTheta[doc][topicMatch[topic]]);
		}
		avgKLD+=kld;
	}
	cout << avgKLD/trainSeqTheta.size() << endl;

	avgKLD=0.0;
	for (int doc=0; doc<testParTheta.size(); doc++)
	{
		double kld=0.0;
		for (int topic=0; topic<numTopics; topic++)
		{
			kld+=testParTheta[doc][topic]*log2(testParTheta[doc][topic]/testSeqTheta[doc][topicMatch[topic]]);
		}
		avgKLD+=kld;
	}
	cout << avgKLD/testSeqTheta.size() << endl;
}
