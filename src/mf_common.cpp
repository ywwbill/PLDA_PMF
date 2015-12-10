//
// Created by Zheng Yan on 12/7/15.
//

#include "mf.h" // everything

using namespace std;

vector<Triplet> train_vec, probe_vec;
vector<double> err_train, err_valid;

/*
	Add two matrix elementwise. Throw exception if their dimensions don't match.
*/
void Add(Mat &res, const Mat &rhs) {
    if (rhs.n != res.n || rhs.m != res.m) {
        throw invalid_argument("matrix dimensions are not consistent when trying add two matrices");
        return;
    }
    for (int i = 0; i < res.n; i++) {
        for (int j = 0; j < res.m; j++)
            res.set(i, j, res.get(i,j)+rhs.get(i,j));
    }
}

/*
	Substraction. Throw exception if their dimensions don't match.
*/
void Minus(Mat &res, const Mat & rhs) {
    if (rhs.n != res.n || rhs.m != res.m) {
        throw invalid_argument("matrix dimensions are not consistent when trying add two matrices");
        return;
    }
    for (int i = 0; i < res.n; i++) {
        for (int j = 0; j < res.m; j++)
            res.set(i, j, res.get(i,j)-rhs.get(i,j));
    }
}

/*
	Multiply by a number.
*/
void Scale(Mat &res, const double &mult) {
    for (int i = 0; i < res.n; i++) {
        for (int j = 0; j < res.m; j++)
            res.set(i, j, res.get(i,j) * mult);
    }
}


/*
	Return the square of x.
*/
template<typename Type>
Type sqr(Type x) {
    return x * x;
}


/*
	Sum of Elementwise square.
*/
double SqrSum(const vector<double> &arr) {
    double s = 0;
    for (int i = 0; i < arr.size(); i++)
            s+=sqr(arr[i]);
    return s;
}

void reset(Mat &arr) {
    for (int i = 0; i < arr.n; i++)
        for (int j = 0; j < arr.m; j++)
            arr.set(i, j, 0);
}

/*
	Sum up the 'cnt' value in a vector of Triplet.
	data_vec: the vector of Triplets to be summed up.
*/
double sum_cnt(vector<Triplet> &data_vec) {
    double sum = 0;
    for (int i = 0; i < data_vec.size(); i++) {
        sum += data_vec[i].cnt;
    }
    return sum;
}


/*
	Load the data from data_file and save in data_vec as triplet. Indices start from 0 (NOT from 1).
	data_file: input file name
	data_vec: vector of Triplets read from the input file
*/
void load_data(string data_file, vector<Triplet> &data_vec, int &num_d, int &num_w, int &cur_doc_id) {

    FILE *pFile;
    pFile = fopen(data_file.c_str(), "r");
    int n, doc_id = cur_doc_id;
    fscanf(pFile, "%d", &n);
    while (fscanf(pFile, "%d", &n) > 0) {
        while (n > 0) {
            int word_id, cnt;
            fscanf(pFile, "%d:%d", &word_id, &cnt);
            data_vec.push_back(Triplet(doc_id, word_id, cnt));

            num_w = max(num_w, word_id + 1);
            n -= cnt;
        }
        doc_id++;
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
double CalcObj(const Mat &D, const Mat &W, const vector<Triplet> &train_vec,
               const int &begin_idx, const int &end_idx, const double &lambda, const double &mean_cnt,
               vector<double> &error, vector<double> &pred_out) {
    double acc_error = 0.0, doc_reg = 0.0, word_reg = 0.0;

    for (int i = begin_idx; i <= end_idx; i++) {
        int doc_id = train_vec[i].doc_id, word_id = train_vec[i].word_id;
        double cnt = train_vec[i].cnt;

        // calculate the prediction: inner product of D(doc_id), W(word_id)
        double predict = 0.0;
        for (int j = 0; j < D.m; j++) {
            double dd = D.get(doc_id,j), ww = W.get(word_id, j);
            predict += (dd * ww);
            doc_reg += sqr(dd);
            word_reg += sqr(ww);
        }
        pred_out[i - begin_idx] = predict + mean_cnt;
        error[i - begin_idx] = pred_out[i - begin_idx] - train_vec[i].cnt;
        // calculate the error between predict and ground truth cnt
        acc_error += sqr(pred_out[i - begin_idx] - train_vec[i].cnt);
    }
    // Objective function value
    double F = acc_error + 0.5 * lambda * (doc_reg + word_reg);

    return F;
}

/*
	Parse arguments passed to main function, return success if train_file and probe_file are found.
*/
bool ParseMainArgs(int argc, char **argv, string &train_file, string &probe_file) {
    if (argc < 3) {
        printf("Please provide train and probe file name e.g. mf train.txt probe.txt\n");
        return false;
    }
    train_file = string(argv[1]);
    probe_file = string(argv[2]);
    return true;
}

bool ParseMainArgs(int argc, char **argv, string &train_file, string &probe_file, string &mode) {
    if (!ParseMainArgs(argc, argv, train_file, probe_file))
        return false;
    mode = string(argv[3]);
    return true;
}