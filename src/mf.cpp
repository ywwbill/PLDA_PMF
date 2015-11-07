/*
This file implements PMF in C++. In this implementation, batch gradient descent is used to minimize loss function: 
F = acc_error + doc_reg + word_reg 
	= sum_{(doc_id, word_id, cnt):train_vec} (cnt - D(doc_id)*W(word_id))^2 + 0.5*lambda*(|D|^2 + |W|^2)

The program should take two data files i.e. training data and testing data.
The input data is in the format of 
n1 w_a(1,1):cnt_a(1,1) w_a(1,2):cnt_a(1,2) ... w_a(1,n1):cnt_a(1,n1)
n2 w_a(2,1):cnt_a(2,1) w_a(2,2):cnt_a(2,2) ... w_a(2,n2):cnt_a(2,n2)
...
nm w_a(m,1):cnt_a(m,1) w_a(m,2):cnt_a(m,2) ... w_a(m,nm):cnt_a(m,nm)


The program will print to the screen the training loss and accuracy on testing data in each epoch and batch. 


Author: Xi Yi (yixi1992@gmail.com)
Date: 11/05/2015
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

#define RANDOM "random"

/*
	Triplet structure where each triplet is {doc_id, word_id, cnt} 
*/
struct Triplet{
	int doc_id, word_id;
	double cnt;
	Triplet(int dd, int ww, double cc):doc_id(dd), word_id(ww), cnt(cc){}
};

/*
	Matrix structure.
*/
class Mat{
	private:
	
	int n, m;
	double **arr;
	
	/*
		Generator who generates initial values for Mat elements.
	*/
	double ValueGenerator(string default_val){
		double val = 0;
		if (default_val==RANDOM){ // if default_val is random, draw a random number.
			val = 0.1 * ((double)rand()) / RAND_MAX;
		} else { // otherwise, default_val is a double value.
			val = std::strtod(default_val.c_str(), NULL);
		}
		return val;
	}
	
	/*
		Convert double to string.
	*/
	string ToString(double d){
		ostringstream strs;
		strs << d;
		return strs.str();
	}
	
	
	/*
		Until c++11, constructor cannot call another constructor so I created this InitMat function to do initialization.
	*/
	void InitMat(int nn, int mm, string default_val=RANDOM) {
		n = nn; m = mm;
		arr = new double*[n];
		for (int i=0; i<n; i++) {
			arr[i] = new double[m];
			for (int j=0; j<m; j++)
				arr[i][j] = ValueGenerator(default_val);
		}
	}
	
	
	public:
	/*
		Constructor for constructing a nn*mm matrix with default_val (double).
	*/
	Mat(int nn, int mm, double default_val){
		InitMat(nn, mm, ToString(default_val));
	}
	
	
	/*
		Constructor for constructing a nn*mm matrix with default_val.
		default_val: a string indicating default values of the matrix which is either RANDOM or any double value, e.g. "0.5"
	*/
	Mat(int nn=0, int mm=0, string default_val = RANDOM) {
		InitMat(nn, mm, default_val);
	}
	
	/*
		Copy constructor. Deep copy.
	*/
	Mat(const Mat &cp){
		InitMat(cp.get_n(),cp.get_m());
		for (int i=0; i<n; i++)
			for (int j=0; j<m; j++)
				arr[i][j] = cp.get(i,j);
	}
	
	int get_n() const {
		return n;
	}
	
	int get_m() const {
		return m;
	}
	
	double get(int i, int j) const {
		if (i>=n || i<0 || j<0 || j>=m) {
			throw invalid_argument("Index out of Bound when using get(i,j) for Mat object.");
		}
		return arr[i][j];
	}
	
	/*
		This class (Proxy) is to enable client use two dimensional subscripts. i.e. Mat a=Mat(1,1,0); a[0][0]+=2;
	*/
	class Proxy {
		public:
			Proxy(double* _array) : _array(_array) { }

			double & operator[](int index) {
				return _array[index];
			}
		private:
			double* _array;
	};
	
	Proxy & operator[](int i){
		if (i>=n || i<0) {
			throw invalid_argument("Index out of Bound when using [] for Mat object.");
		}
		return *(new Proxy(arr[i]));
	}
	
	
	/*
		Add two matrix elementwise. Throw exception if their dimensions don't match.
	*/
	Mat operator +(const Mat& rhs){
		if (rhs.get_n()!=n || rhs.get_m()!=m){
			throw invalid_argument("Matrix dimensions are not consistent when trying '+' two matrices");
			return *this;
		}
		Mat res = Mat(*this);
		for (int i=0; i<res.get_n(); i++){
			for (int j=0; j<res.get_m(); j++)
				res[i][j] += rhs.get(i,j);
		}
		return res;
	}
	
	/*
		Substraction. Throw exception if their dimensions don't match.
	*/
	Mat operator -(const Mat& rhs){
		if (rhs.get_n()!=n || rhs.get_m()!=m){
			throw invalid_argument("Matrix dimensions are not consistent when trying '+' two matrices");
			return Mat();
		}
		Mat res = Mat(*this);
		for (int i=0; i<res.get_n(); i++){
			for (int j=0; j<res.get_m(); j++)
				res[i][j] -= rhs.get(i, j);
		}
		return res;
	}
	
	/*
		Multiply by a number.
	*/
	Mat operator *(const double& mult){
		Mat res = Mat(*this);
		for (int i=0; i<res.get_n(); i++){
			for (int j=0; j<res.get_m(); j++)
				res[i][j] *= mult;
		}
		return res;
	}
	
	/*
		Return a matrix which is the elementwise square.
	*/
	Mat Sqr(){
		Mat res = Mat(*this);
		for (int i=0; i<res.get_n(); i++)
			for (int j=0; j<res.get_m(); j++)
				res[i][j] = res[i][j]*res[i][j];
		return res;
	}
	
	/*
		Return the sum of all elements in the matrix.
	*/
	double Sum(){
		double s;
		for (int i=0; i<n; i++)
			for (int j=0; j<m; j++)
				s += arr[i][j];
		return s;
	}
	
};

vector<Triplet> train_vec, probe_vec;
vector<double> err_train, err_valid;

/*
	Load the data from data_file and save in data_vec as triplet. Indices start from 0 (NOT from 1).
	data_file: input file name
	data_vec: vector of Triplets read from the input file
*/
void load_data(string data_file, vector<Triplet> &data_vec, int &num_d, int &num_w){
	FILE *pFile;
	pFile = fopen(data_file.c_str(), "r");
	int doc_id = 0, n;
	while (fscanf(pFile, "%d", &n)>0){
		for (int i=0; i<n; i++){
			int word_id;
			double cnt;
			fscanf(pFile, "%d:%lf", &word_id, &cnt);
			word_id--;
			if (word_id + 1 > num_w) num_w = word_id + 1;
			data_vec.push_back(Triplet(doc_id, word_id, cnt));
		}
		doc_id ++;
	}
	num_d = doc_id;
	fclose(pFile);
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
	Return the square of x.
*/
template <typename Type>
Type sqr(Type x)
{
   return x * x;
}

/*
	Objective function value calculator:
	Takes Doc feature matrix and Word feature matrix, as well as ground truth counts,
	return objective function value and a vector of error value for each case.
	The objective function is 
		F = acc_error + doc_reg + word_reg 
			= sum_{(doc_id, word_id, cnt):train_vec} (cnt - D(doc_id)*W(word_id))^2 + 0.5*lambda*(|D|^2 + |W|^2)
*/
double CalcObj(const Mat &D, const Mat &W, const vector<Triplet> &train_vec, const int &begin_idx, const int &end_idx, const double &lambda, const double &mean_cnt, Mat &error, Mat &pred_out){
	double acc_error = 0.0, doc_reg = 0.0, word_reg = 0.0;
	error = Mat(end_idx-begin_idx+1, 1, 0); pred_out = Mat(end_idx-begin_idx+1, 1, 0);
	for (int i = begin_idx; i <= end_idx; i++){
		int doc_id = train_vec[i].doc_id, word_id = train_vec[i].word_id;
		double cnt = train_vec[i].cnt;
		
		// calculate the prediction: inner product of D(doc_id), W(word_id)
		double predict = 0.0;
		for (int j=0; j < D.get_m(); j++){
			double dd = D.get(doc_id, j), ww = W.get(word_id, j);
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
		printf("Please provide train and probe file name e.g. mf train.txt probe\n");
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
	
	
	double epsilon = 50.0; // Learning rate 
	double lambda = 0.01; // Regularization parameter 
	double momentum = 0.8;
	
	
	int epoch = 0;
	int maxepoch = 50;
	
	int num_d = 1;  // Number of docs 
	int num_w = 2;  // Number of words 
	int num_feat = 1; // Rank 10 decomposition 
	int tmp_num_d, tmp_num_w;
	
	load_data(train_file, train_vec, num_d, num_w); // Triplets: {doc_id, word_id, cnt} 
	load_data(probe_file, probe_vec, tmp_num_d, tmp_num_w);
	
	int pairs_tr = train_vec.size(); // training data
	int pairs_pr = probe_vec.size(); // validation data
	
	double mean_cnt = sum_cnt(train_vec)/pairs_tr;
	
	int numbatches = 1; // Number of batches  
	int N = (pairs_tr-1)/numbatches+1; // number training triplets per batch 
	
	Mat D = Mat(num_d, num_feat); // Doc feature vectors
	Mat W = Mat(num_w, num_feat); // Word feature vecators
	Mat D_inc = Mat(num_d, num_feat, 0);
	Mat W_inc = Mat(num_w, num_feat, 0);
	Mat error, pred_out;

	double F = 0.0;
	
	for (; epoch < maxepoch; epoch++){
		// Random permute training data.
		std::random_shuffle ( train_vec.begin(), train_vec.end() );
		
		for (int batch = 1; batch <= numbatches; batch++ ){
		
			int begin_idx = (batch-1)*N, end_idx = min(batch*N-1, pairs_tr-1), N = end_idx - begin_idx +1;

			//%%%%%%%%%%%%%% Compute Predictions %%%%%%%%%%%%%%%%%
			F = CalcObj(D, W, train_vec, begin_idx, end_idx, lambda, mean_cnt, error, pred_out);
			

			//%%%%%%%%%%%%%% Compute Gradients %%%%%%%%%%%%%%%%%%%
			Mat Ix_D = Mat(N, num_feat, 0), Ix_W = Mat(N, num_feat, 0);
			for (int i = begin_idx; i <= end_idx; i++){
				int doc_id = train_vec[i].doc_id, word_id = train_vec[i].word_id;				
				for (int j=0; j < num_feat; j++){
					double dd = D[doc_id][j], ww = W[word_id][j];
					Ix_D[i - begin_idx][j] = error[i-begin_idx][0] *2* ww + lambda*dd;
					Ix_W[i - begin_idx][j] = error[i-begin_idx][0] *2* dd + lambda*ww;
				}
			}
			
			Mat d_D = Mat(num_d, num_feat, 0);
			Mat d_W = Mat(num_w, num_feat, 0);

			for (int i = begin_idx; i <= end_idx; i++){
				int doc_id = train_vec[i].doc_id, word_id = train_vec[i].word_id;
				for (int j=0; j<num_feat; j++){
					d_D[doc_id][j] += Ix_D[i-begin_idx][j];
					d_W[word_id][j] += Ix_W[i-begin_idx][j];
				}
			}

			//%%%% Update doc and word features %%%%%%%%%%%

			D_inc = D_inc*momentum + d_D*(epsilon/N);
			D =  D - D_inc;

			W_inc = W_inc*momentum + d_W*(epsilon/N);
			W =  W - W_inc;
		
			//%%%%%%%%%%%%%% Compute Predictions after Parameter Updates %%%%%%%%%%%%%%%%%
			F = CalcObj(D, W, train_vec, begin_idx, end_idx, lambda, mean_cnt, error, pred_out);
			err_train.push_back(sqrt(F/N));

			//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			//%%% Compute predictions on the validation set %%%%%%%%%%%%%%%%%%%%%% 
			F = CalcObj(D, W, probe_vec, 0, pairs_pr-1, lambda, mean_cnt, error, pred_out);
			err_valid.push_back(sqrt(error.Sqr().Sum())/pairs_pr);
			printf("epoch %4i batch %4i Training RMSE %6.4f  Test RMSE %6.4f  \n", epoch, batch, err_train[epoch], err_valid[epoch]);
		}
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	}
}