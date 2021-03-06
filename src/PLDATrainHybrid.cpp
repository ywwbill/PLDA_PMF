#include <stdlib.h>
#include <memory.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "mpi.h"
#include <omp.h>
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

int main(int argc, char** argv) //argv[1]: corpusFileName  argv[2]: thetaFileName  argv[3]: phiFileName
{
	int numTopics=atoi(argv[4]);
	int numVocab=atoi(argv[5]);
	int numIters=atoi(argv[6]);
	double alpha=atof(argv[7]);
	double beta=atof(argv[8]);

//	cout << "#topics: " << numTopics << endl;
//	cout << "#vocab: " << numVocab << endl;
//	cout << "#iters: " << numIters << endl;
//	cout << "alpha: " << alpha << endl;
//	cout << "beta: " << beta << endl;

	int rank,numThreads;
	MPI_Init(&argc, &argv);
	double startTime=MPI_Wtime();
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numThreads);
	MPI_Status status;

	//read corpus
	int numGlobalDocs,docOnNode[numThreads];
	string line,data="";
	if (rank==0)
	{
		ifstream fin(argv[1]);
		fin >> numGlobalDocs;
		getline(fin, line);

		//read its own docs
		int startDoc=0,endDoc=numGlobalDocs/numThreads-1;
		int tempLocalDocs=endDoc-startDoc+1;
		docOnNode[0]=tempLocalDocs;
		string tempData="";
		for (int i=0; i<tempLocalDocs; i++)
		{
			getline(fin, line);
			if (isspace(line[line.length()-1]))
			{
				line.erase(line.length()-1);
			}
			if (data.length()>0)
			{
				data=data+"#";
			}
			data=data+line;
		}

		//read and send docs for other nodes
		for (int i=1; i<numThreads; i++)
		{
			startDoc=i*numGlobalDocs/numThreads;
			endDoc=(i+1)*numGlobalDocs/numThreads-1;
			if (i==numThreads-1)
			{
				endDoc=numGlobalDocs-1;
			}
			tempLocalDocs=endDoc-startDoc+1;
			docOnNode[i]=tempLocalDocs;
			tempData="";
			for (int j=0; j<tempLocalDocs; j++)
			{
				getline(fin, line);
				if (isspace(line[line.length()-1]))
				{
					line.erase(line.length()-1);
				}
				if (tempData.length()>0)
				{
					tempData+="#";
				}
				tempData+=line;
			}
			MPI_Send((char*)tempData.c_str(), tempData.length(), MPI_CHAR, i, 0, MPI_COMM_WORLD);
		}

		fin.close();
	}
	else
	{
		//get the docs from node 0
		MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
		int len;
		MPI_Get_count(&status, MPI_CHAR, &len);
		char *buf=new char[len];
		MPI_Recv(buf, len, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
		string tempData(buf);
		data=tempData;
	}

//	cout << "Rank " << rank << ": " << data << endl;


	//get string corpus
	vector<string> strCorpus;
	split(data, "#", strCorpus);
	int numLocalDocs=strCorpus.size();

	//convert to int corpus
	vector< vector<int> > corpus;
	vector<string> seg,segseg;
	int numLocalWords=0;
	int word,count;
	for (int i=0; i<strCorpus.size(); i++)
	{
		split(strCorpus[i], " ", seg);
		numLocalWords+=atoi(seg[0].c_str());

		vector<int> doc;
		for (int j=1; j<seg.size(); j++)
		{
			split(seg[j], ":", segseg);
			word=atoi(segseg[0].c_str());
			count=atoi(segseg[1].c_str());
			for (int k=0; k<count; k++)
			{
				doc.push_back(word);
			}
		}
		corpus.push_back(doc);
	}

	//get total #words
	int numGlobalWords=0;
	if (rank==0)
	{
		numGlobalWords=numLocalWords;
		int tempLocalWords;
		for (int i=1; i<numThreads; i++)
		{
			MPI_Recv(&tempLocalWords, 1, MPI_INT, i, i, MPI_COMM_WORLD, &status);
			numGlobalWords+=tempLocalWords;
		}
//		cout << numWords << endl;
	}
	else
	{
		MPI_Send(&numLocalWords, 1, MPI_INT, 0, rank, MPI_COMM_WORLD);
	}

	//init
	int docTopicCounts[numLocalDocs][numTopics];
	int localTopicVocabCounts[numTopics][numVocab];
	int localTopicTokenCounts[numTopics];
	memset(docTopicCounts, 0, sizeof(docTopicCounts));
	memset(localTopicVocabCounts, 0, sizeof(localTopicVocabCounts));
	memset(localTopicTokenCounts, 0, sizeof(localTopicTokenCounts));

	vector< vector<int> > topicAssigns;
	srand((unsigned)time(NULL));
	for (int doc=0; doc<numLocalDocs; doc++)
	{
		vector<int> assign;
		for (int i=0; i<corpus[doc].size(); i++)
		{
			word=corpus[doc][i];
			int topic=rand()%numTopics;
			assign.push_back(topic);
			docTopicCounts[doc][topic]++;
			localTopicVocabCounts[topic][word]++;
			localTopicTokenCounts[topic]++;
		}
		topicAssigns.push_back(assign);
	}

	//get GlobalTopicVocabCounts and GlobalTopicTokenCounts
	int globalTopicVocabCounts[numTopics][numVocab];
	int globalTopicTokenCounts[numTopics];
	memset(globalTopicVocabCounts, 0, sizeof(globalTopicVocabCounts));
	memset(globalTopicTokenCounts, 0, sizeof(globalTopicTokenCounts));
	if (rank==0)
	{
		for (int topic=0; topic<numTopics; topic++)
		{
			for (int vocab=0; vocab<numVocab; vocab++)
			{
				globalTopicVocabCounts[topic][vocab]=localTopicVocabCounts[topic][vocab];
				globalTopicTokenCounts[topic]+=localTopicVocabCounts[topic][vocab];
			}
		}

		//collect localTopicVocabCounts and compute globalTopicVocabCounts & globalTopicTokenCounts
		int tempTopicVocabCounts[numTopics][numVocab];
		for (int i=1; i<numThreads; i++)
		{
			MPI_Recv(&(tempTopicVocabCounts[0][0]), numTopics*numVocab, MPI_INT, i, i, MPI_COMM_WORLD, &status);
			for (int topic=0; topic<numTopics; topic++)
			{
				for (int vocab=0; vocab<numVocab; vocab++)
				{
					globalTopicVocabCounts[topic][vocab]+=tempTopicVocabCounts[topic][vocab];
					globalTopicTokenCounts[topic]+=tempTopicVocabCounts[topic][vocab];
				}
			}
		}

		//send globalTopicVocabCounts
		for (int i=1; i<numThreads; i++)
		{
			MPI_Send(&(globalTopicVocabCounts[0][0]), numTopics*numVocab, MPI_INT, i, 0, MPI_COMM_WORLD);
		}
	}
	else
	{
		//Send localTopicVocabCounts
		MPI_Send(&(localTopicVocabCounts[0][0]), numTopics*numVocab, MPI_INT, 0, rank, MPI_COMM_WORLD);
		//Recv globalTopicVocabCounts and compute globalTopicTokenCounts
		MPI_Recv(&(globalTopicVocabCounts[0][0]), numTopics*numVocab, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		for (int topic=0; topic<numTopics; topic++)
		{
			for (int vocab=0; vocab<numVocab; vocab++)
			{
				globalTopicTokenCounts[topic]+=globalTopicVocabCounts[topic][vocab];
			}
		}
	}

//	cout << "Rank " << rank << endl;
//	for (int topic=0; topic<numTopics; topic++)
//	{
//		cout << localTopicTokenCounts[topic] << " ";
//	}
//	cout << endl;
//	for (int topic=0; topic<numTopics; topic++)
//	{
//		cout << globalTopicTokenCounts[topic] << " ";
//	}
//	cout << endl;

	MPI_Barrier(MPI_COMM_WORLD);

	int oldTopic,newTopic;
	double topicScores[numTopics],sum,sample;
	double localLogLLD,globalLogLLD,perplexity;
	double localTheta[numLocalDocs][numTopics];
	double phi[numTopics][numVocab];
	for (int iter=1; iter<=numIters; iter++)
	{
		//sample each doc
		for (int doc=0; doc<numLocalDocs; doc++)
		{
			//sample each topic assignment for each token
			for (int token=0; token<corpus[doc].size(); token++)
			{
				//remove old topic assignment
				word=corpus[doc][token];
				oldTopic=topicAssigns[doc][token];
				docTopicCounts[doc][oldTopic]--;
				localTopicVocabCounts[oldTopic][word]--;
				localTopicTokenCounts[oldTopic]--;
				globalTopicVocabCounts[oldTopic][word]--;
				globalTopicTokenCounts[oldTopic]--;

				//compute the probability of each topic
				sum=0.0;
				for (int topic=0; topic<numTopics; topic++)
				{
					topicScores[topic]=(docTopicCounts[doc][topic]+alpha)*
							(globalTopicVocabCounts[topic][word]+beta)/(globalTopicTokenCounts[topic]+numVocab*beta);
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
				localTopicVocabCounts[newTopic][word]++;
				localTopicTokenCounts[newTopic]++;
				globalTopicVocabCounts[newTopic][word]++;
				globalTopicTokenCounts[newTopic]++;
			}
		}


		if(iter%5!=0) continue;	
		
		
		//compute theta
		#pragma omp parallel for shared(localTheta) schedule(dynamic)
		for (int doc=0; doc<numLocalDocs; doc++)
		{
			for (int topic=0; topic<numTopics; topic++)
			{
				localTheta[doc][topic]=(docTopicCounts[doc][topic]+alpha)/(corpus[doc].size()+numTopics*alpha);
			}
		}

		memset(globalTopicVocabCounts, 0, sizeof(globalTopicVocabCounts));
		memset(globalTopicTokenCounts, 0, sizeof(globalTopicTokenCounts));
		if (rank==0)
		{
			for (int topic=0; topic<numTopics; topic++)
			{
				for (int vocab=0; vocab<numVocab; vocab++)
				{
					globalTopicVocabCounts[topic][vocab]=localTopicVocabCounts[topic][vocab];
					globalTopicTokenCounts[topic]+=localTopicVocabCounts[topic][vocab];
				}
			}

			//collect localTopicVocabCounts and compute globalTopicVocabCounts & globalTopicTokenCounts
			int tempTopicVocabCounts[numTopics][numVocab];
			for (int i=1; i<numThreads; i++)
			{
				MPI_Recv(&(tempTopicVocabCounts[0][0]), numTopics*numVocab, MPI_INT, i, i, MPI_COMM_WORLD, &status);
				for (int topic=0; topic<numTopics; topic++)
				{
					for (int vocab=0; vocab<numVocab; vocab++)
					{
						globalTopicVocabCounts[topic][vocab]+=tempTopicVocabCounts[topic][vocab];
						globalTopicTokenCounts[topic]+=tempTopicVocabCounts[topic][vocab];
					}
				}
			}

			//send globalTopicVocabCounts
			for (int i=1; i<numThreads; i++)
			{
				MPI_Send(&(globalTopicVocabCounts[0][0]), numTopics*numVocab, MPI_INT, i, 0, MPI_COMM_WORLD);
			}
		}
		else
		{
			//Send localTopicVocabCounts
			MPI_Send(&(localTopicVocabCounts[0][0]), numTopics*numVocab, MPI_INT, 0, rank, MPI_COMM_WORLD);
			//Recv globalTopicVocabCounts and compute globalTopicTokenCounts
			MPI_Recv(&(globalTopicVocabCounts[0][0]), numTopics*numVocab, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
			for (int topic=0; topic<numTopics; topic++)
			{
				for (int vocab=0; vocab<numVocab; vocab++)
				{
					globalTopicTokenCounts[topic]+=globalTopicVocabCounts[topic][vocab];
				}
			}
		}

		//compute phi
		#pragma omp parallel for shared(phi) schedule(dynamic)
		for (int topic=0; topic<numTopics; topic++)
		{
			for (int vocab=0; vocab<numVocab; vocab++)
			{
				phi[topic][vocab]=(globalTopicVocabCounts[topic][vocab]+beta)/(globalTopicTokenCounts[topic]+numVocab*beta);
			}
		}

		//compute local LogLLD
		localLogLLD=0.0;
		#pragma omp parallel for private(word,sum) reduction(+:localLogLLD) schedule(dynamic)
		for (int doc=0; doc<numLocalDocs; doc++)
		{
			for (int token=0; token<corpus[doc].size(); token++)
			{
				word=corpus[doc][token];
				sum=0.0;
				for (int topic=0; topic<numTopics; topic++)
				{
					sum+=localTheta[doc][topic]*phi[topic][word];
				}
				localLogLLD+=log(sum);
			}
		}

		//compute globalLogLLD and perplexity
		if (rank==0)
		{
			globalLogLLD=localLogLLD;
			double tempLogLLD;
			for (int i=1; i<numThreads; i++)
			{
				MPI_Recv(&tempLogLLD, 1, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);
				globalLogLLD+=tempLogLLD;
			}
			perplexity=exp(-globalLogLLD/numGlobalWords);
			cout << "<" << iter << ">\tLog-LLD: " << globalLogLLD << "\tPPX: " << perplexity << endl;
		}
		else
		{
			MPI_Send(&localLogLLD, 1, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}

	if (rank==0)
	{
		ofstream foutTheta(argv[2]);
		for (int doc=0; doc<numLocalDocs; doc++)
		{
			for (int topic=0; topic<numTopics; topic++)
			{
				foutTheta << localTheta[doc][topic] << " ";
			}
			foutTheta << endl;
		}
		for (int i=1; i<numThreads; i++)
		{
			double tempTheta[docOnNode[i]][numTopics];
			MPI_Recv(&(tempTheta[0][0]), docOnNode[i]*numTopics, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);
			for (int doc=0; doc<docOnNode[i]; doc++)
			{
				for (int topic=0; topic<numTopics; topic++)
				{
					foutTheta << tempTheta[doc][topic] << " ";
				}
				foutTheta << endl;
			}
		}
		foutTheta.close();

		ofstream foutPhi(argv[3]);
		for (int topic=0; topic<numTopics; topic++)
		{
			for (int vocab=0; vocab<numVocab; vocab++)
			{
				foutPhi << phi[topic][vocab] << " ";
			}
			foutPhi << endl;
		}
		foutPhi.close();
	}
	else
	{
		MPI_Send(&(localTheta[0][0]), numLocalDocs*numTopics, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
	}

	double endTime=MPI_Wtime();
	cout << "Thread " << rank << " compute time: " << endTime-startTime << endl;

	MPI_Finalize();
	return 0;
}
