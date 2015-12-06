/*
This file implements PMF in C++. In this implementation, batch gradient descent is used to minimize loss function: 
F = acc_error + doc_reg + word_reg 
	= sum_{(doc_id, word_id, cnt):train_vec} (cnt - D(doc_id)*W(word_id))^2 + 0.5*lambda*(|D|^2 + |W|^2)

The program should take two data files i.e. training data and testing data.
The input data is in the forvector<vector<double> > of 
total_word_in_doc1 w_a(1,1):cnt_a(1,1) w_a(1,2):cnt_a(1,2) ... w_a(1,n1):cnt_a(1,n1)
total_word_in_doc2 w_a(2,1):cnt_a(2,1) w_a(2,2):cnt_a(2,2) ... w_a(2,n2):cnt_a(2,n2)
...
total_word_in_docm w_a(m,1):cnt_a(m,1) w_a(m,2):cnt_a(m,2) ... w_a(m,nm):cnt_a(m,nm)

where total_word_in_dock = \sum_i cnt_a(k,i)

The program will print to the screen the training loss and accuracy on testing data in each epoch and batch. 

Usage:
g++ src/mf.cpp -o mf
./mf data/train_corpus.txt data/test_corpus.txt

Author: Xi Yi (yixi1992@gmail.com)
Date: 12/05/2015
*/

#include <stdio.h> // fscanf
#include <stdlib.h> // std::rand, std::srand
#include <string.h> // strlen, strcmp
#include <algorithm> // std::random_shuffle, min
#include <vector> // std::vector
#include <stdexcept> // throw
#include <sstream> // ostringstream
#include <math.h> //sqrt
using namespace std;

#define RANDOM -1

/*
	Triplet structure where each triplet is {doc_id, word_id, cnt} 
*/
struct Triplet{
	int doc_id, word_id;
	double cnt;
	Triplet(int dd, int ww, double cc):doc_id(dd), word_id(ww), cnt(cc){}
};

vector<Triplet> train_vec, probe_vec;
vector<double> err_train, err_valid;


vector<vector<double> > NewArray(int n, int m, int default_val){	
	vector<vector<double> > a(n, vector<double>(m));
	if (default_val==RANDOM){
		for (int i=0; i<a.size(); i++)
			for (int j=0; j<a[0].size(); j++)
				a[i][j] = 0.1 * ((double)rand()) / RAND_MAX;
	}
	return a;
	
}

/*
	Add two matrix elementwise. Throw exception if their dimensions don't match.
*/
void Add(vector<vector<double> > & res, const vector<vector<double> > lhs, const vector<vector<double> > rhs){
	if (rhs.size()!=lhs.size() || rhs[0].size()!=lhs[0].size()){
		throw invalid_argument("matrix dimensions are not consistent when trying add two matrices");
		return;
	}
	res.clear();
	res = NewArray(lhs.size(), lhs[0].size(), 0);
	for (int i=0; i<lhs.size(); i++){
		for (int j=0; j<lhs[0].size(); j++)
			res[i][j] = lhs[i][j]+rhs[i][j];
	}
}

/*
	Substraction. Throw exception if their dimensions don't match.
*/
void Minus(vector<vector<double> > & res, const vector<vector<double> > lhs, const vector<vector<double> > rhs){
	if (rhs.size()!=lhs.size() || rhs[0].size()!=lhs[0].size()){
		throw invalid_argument("matrix dimensions are not consistent when trying add two matrices");
		return;
	}
	res.clear();
	res = NewArray(lhs.size(), lhs[0].size(), 0);
	for (int i=0; i<lhs.size(); i++){
		for (int j=0; j<lhs[0].size(); j++)
			res[i][j] = lhs[i][j]-rhs[i][j];
	}
}

/*
	Multiply by a number.
*/
void Scale(vector<vector<double> > & res, const vector<vector<double> > lhs, const double& mult){
	res.clear();
	res = NewArray(lhs.size(), lhs[0].size(), 0);
	for (int i=0; i<lhs.size(); i++){
		for (int j=0; j<lhs[0].size(); j++)
			res[i][j] = lhs[i][j]*mult;
	}
}


/*
	Return the square of x.
*/
template <typename Type>
Type sqr(Type x)
{
   return x * x;
}


/*
	Elementwise square the matrix.
*/
vector<vector<double> > Sqr(const vector<vector<double> > & arr){
	vector<vector<double> > res = NewArray(arr.size(), arr[0].size(), 0);
	for (int i=0; i<arr.size(); i++)
		for (int j=0; j<arr[0].size(); j++)
			res[i][j] = sqr(arr[i][j]);
	return res;
}

/*
	Return the sum of all elements in the matrix.
*/
double Sum(const vector<vector<double> > &arr){
	double s;
	for (int i=0; i<arr.size(); i++)
		for (int j=0; j<arr[0].size(); j++)
			s += arr[i][j];
	return s;
}

void reset(vector<vector<double> > &arr){
	for (int i=0; i<arr.size(); i++)
		for (int j=0; j<arr[0].size(); j++)
			arr[i][j] = 0;
}

/*
	Sum up the 'cnt' value in a vector of Triplet.
	data_vec: the vector of Triplets to be summed up.
*/
double sum_cnt(vector<Triplet> &data_vec){
	double sum = 0;
	for (int i=0; i<data_vec.size(); i++){
		sum += data_vec[i].cnt;
	}
	return sum;
}


/*
	Load the data from data_file and save in data_vec as triplet. Indices start from 0 (NOT from 1).
	data_file: input file name
	data_vec: vector of Triplets read from the input file
*/
void load_data(string data_file, vector<Triplet> &data_vec, int &num_d, int &num_w, int &cur_doc_id){

	FILE *pFile;
	pFile = fopen(data_file.c_str(), "r");
	int n, doc_id=cur_doc_id;
	fscanf(pFile, "%d", &n);
	while (fscanf(pFile, "%d", &n)>0){
		while (n>0){	
			int word_id, cnt;
			fscanf(pFile, "%d:%d", &word_id, &cnt);
			data_vec.push_back(Triplet(doc_id, word_id, cnt));
			
			num_w = max(num_w, word_id + 1);
			n-=cnt;
		}
		doc_id ++;
	}
	num_d = doc_id - cur_doc_id;
	cur_doc_id = doc_id;
	fclose(pFile);
}


/*
	Objective function value calculator:
	Takes Doc feature matrix and Word feature matrix, as well as ground truth counts,
	return objective function value and a vector of error value for each case.
	The objective function is 
		F = acc_error + doc_reg + word_reg 
			= sum_{(doc_id, word_id, cnt):train_vec} (cnt - D(doc_id)*W(word_id))^2 + 0.5*lambda*(|D|^2 + |W|^2)
*/
double CalcObj(const vector<vector<double> > &D, const vector<vector<double> > &W, const vector<Triplet> &train_vec, const int &begin_idx, const int &end_idx, const double &lambda, const double &mean_cnt, vector<vector<double> > &error, vector<vector<double> > &pred_out){
	double acc_error = 0.0, doc_reg = 0.0, word_reg = 0.0;
	for (int i = begin_idx; i <= end_idx; i++){
		int doc_id = train_vec[i].doc_id, word_id = train_vec[i].word_id;
		double cnt = train_vec[i].cnt;
		
		// calculate the prediction: inner product of D(doc_id), W(word_id)
		double predict = 0.0;
		for (int j=0; j < D[0].size(); j++){
			double dd = D[doc_id][j], ww = W[word_id][j];
			predict += (dd*ww);
			doc_reg += sqr(dd);
			word_reg += sqr(ww);
		}
		pred_out[i-begin_idx][0] = predict + mean_cnt;
		error[i - begin_idx ][0] = pred_out[i-begin_idx][0] - train_vec[i].cnt;
		// calculate the error between predict and ground truth cnt
		acc_error += sqr(pred_out[i-begin_idx][0] - train_vec[i].cnt);
	}
	// Objective function value
	double F = acc_error + 0.5*lambda*(doc_reg + word_reg);

	return F;
}

/*
	Parse arguments passed to main function, return success if train_file and probe_file are found.
*/
bool ParseMainArgs(int argc, char **argv, string &train_file, string &probe_file){
	if (argc<3){
		printf("Please provide train and probe file name e.g. mf train.txt probe.txt\n");
		return false;
	}
	train_file = string(argv[1]);
	probe_file = string(argv[2]);
	return true;
}

/*
	main
*/
int main(int argc, char **argv){
	string train_file = "";
	string probe_file = "";
	
	if (!ParseMainArgs(argc, argv, train_file, probe_file)) return 0;
	
	
	double epsilon = 5; // Learning rate
	double lambda = 1; // Regularization parameter 
	double momentum = 0.9;
	
	
	int epoch = 0;
	int maxepoch = 50;
	
	int num_d = 2020;  // Number of docs
	int num_w = 7167;  // Number of words
	int num_feat = 10; // Rank 10 decomposition
	int tmp_num_d;
	int cur_doc_id = 0;
	
	vector<Triplet> all_data_vec;
	load_data(train_file, all_data_vec, tmp_num_d, num_w, cur_doc_id); // Triplets: {doc_id, word_id, cnt} 
	load_data(probe_file, all_data_vec, tmp_num_d, num_w, cur_doc_id);
	num_d = cur_doc_id;
	
	for (int i=0; i<all_data_vec.size(); i++){
		if (rand()%100<90) train_vec.push_back(all_data_vec[i]);
		else probe_vec.push_back(all_data_vec[i]);
	}
	all_data_vec.clear();
	
	int pairs_tr = train_vec.size(); // training data
	int pairs_pr = probe_vec.size(); // validation data
	
	double mean_cnt = sum_cnt(train_vec)/pairs_tr;
	
	int numbatches = 10; // Number of batches  
	int NN = (pairs_tr-1)/numbatches+1; // number training triplets per batch 
	
	vector<vector<double> > D = NewArray(num_d, num_feat, RANDOM); // Doc feature vectors
	vector<vector<double> > W = NewArray(num_w, num_feat, 0);// Word feature vecators
	vector<vector<double> > D_inc = NewArray(num_d, num_feat, 0), W_inc = NewArray(num_w, num_feat, 0);
	vector<vector<double> > error = NewArray(NN, 1, 0), pred_out = NewArray(NN, 1, 0), test_error = NewArray(pairs_pr, 1, 0), test_pred_out = NewArray(pairs_pr, 1, 0);
	
	vector<vector<double> > Ix_D = NewArray(NN, num_feat, 0), Ix_W = NewArray(NN, num_feat, 0); // gradient w.r.t. training sample
	vector<vector<double> > d_D = NewArray(num_d, num_feat, 0), d_W = NewArray(num_w, num_feat, 0);

	double F = 0.0;
	
	for (; epoch < maxepoch; epoch++){
		// Random permute training data.
		std::random_shuffle ( train_vec.begin(), train_vec.end() );
		
		for (int batch = 1; batch <= numbatches; batch++ ){
		
			int begin_idx = (batch-1)*NN, end_idx = min(batch*NN-1, pairs_tr-1), N = end_idx - begin_idx +1;
			
			//%%%%%%%%%%%%%% Compute Predictions %%%%%%%%%%%%%%%%%
			F = CalcObj(D, W, train_vec, begin_idx, end_idx, lambda, mean_cnt, error, pred_out);
			
			//%%%%%%%%%%%%%% Compute Gradients %%%%%%%%%%%%%%%%%%%
			for (int i = begin_idx; i <= end_idx; i++){
				int doc_id = train_vec[i].doc_id, word_id = train_vec[i].word_id;	
				for (int j=0; j < num_feat; j++){
					double dd = D[doc_id][j], ww = W[word_id][j];
					Ix_D[i - begin_idx][j] = error[i-begin_idx][0] *2* ww + lambda*dd;
					Ix_W[i - begin_idx][j] = error[i-begin_idx][0] *2* dd + lambda*ww;
				}
			}
			
			reset(d_D);
			reset(d_W);
			for (int i = begin_idx; i <= end_idx; i++){
				int doc_id = train_vec[i].doc_id, word_id = train_vec[i].word_id;
				for (int j=0; j<num_feat; j++){
					d_D[doc_id][j] += Ix_D[i-begin_idx][j];
					d_W[word_id][j] += Ix_W[i-begin_idx][j];
				}
			}

			//%%%% Update doc and word features %%%%%%%%%%%

			Scale(D_inc, D_inc, momentum);
			Scale(d_D, d_D, epsilon/N);
			Add(D_inc, D_inc, d_D);
			//D_inc = D_inc*momentum + d_D*(epsilon/N);
			Minus(D, D, D_inc);
			//D =  D - D_inc;

			Scale(W_inc, W_inc, momentum);
			Scale(d_W, d_W, epsilon/N);
			Add(W_inc, W_inc, d_W);
			//W_inc = W_inc*momentum + d_W*(epsilon/N);
			Minus(W, W, W_inc);
			//W =  W - W_inc;
		
			//%%%%%%%%%%%%%% Compute Predictions after Parameter Updates %%%%%%%%%%%%%%%%%
			F = CalcObj(D, W, train_vec, begin_idx, end_idx, lambda, mean_cnt, error, pred_out);
			err_train.push_back(sqrt(F/N));

			//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			//%%% Compute predictions on the validation set %%%%%%%%%%%%%%%%%%%%%% 
			F = CalcObj(D, W, probe_vec, 0, pairs_pr-1, lambda, mean_cnt, test_error, test_pred_out);
			err_valid.push_back(sqrt(Sum(Sqr(test_error))/pairs_pr));
			printf("epoch %4i batch %4i Training RMSE %6.4f  Test RMSE %6.4f  \n", epoch, batch, err_train.back(), err_valid.back());
		}
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	}
}
