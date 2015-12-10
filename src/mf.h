//
// Created by Zheng Yan on 12/6/15.
//

#ifndef PLDA_PMF_MF_H
#define PLDA_PMF_MF_H

#include <stdio.h> // fscanf
#include <stdlib.h> // std::rand, std::srand
#include <string.h> // strlen, strcmp
#include <algorithm> // std::random_shuffle, min
#include <vector> // std::vector
#include <stdexcept> // throw
#include <sstream> // ostringstream

#include <string> // string
#include <vector> // vector
#include <math.h> // sqrt

#define RANDOM -1

/*
	Triplet structure where each triplet is {doc_id, word_id, cnt}
*/
typedef struct Triplet {
    int doc_id, word_id;
    double cnt;

    Triplet() : doc_id(-1), word_id(-1), cnt(0.0) {}
    Triplet(int dd, int ww, double cc) : doc_id(dd), word_id(ww), cnt(cc) { }
} TripletType;

typedef struct Mat {
    int n,m;
    double *arr;
    Mat(){}
    Mat(int nn, int mm, double def): n(nn), m(mm){
        arr = new double[n*m];
        for (int i=0; i<n*m; i++)
            if (def==RANDOM) arr[i] = 0.1 * ((double) rand()) / RAND_MAX;
            else arr[i] = def;
    }

    double get(int i, int j) const{
        return arr[i*m+j];
    }

    void set(int i, int j, double v){
        arr[i*m+j] = v;
    }

    int size() {
        return n * m;
    }
} Mat;

bool ParseMainArgs(int argc, char **argv, std::string &train_file, std::string &probe_file);

void load_data(std::string data_file, std::vector<Triplet> &data_vec, int &num_d, int &num_w, int &cur_doc_id);

double sum_cnt(std::vector<Triplet> &data_vec);

std::vector<std::vector<double> > NewArray(int n, int m, int default_val);

double CalcObj(const Mat &D, const Mat &W,
               const std::vector<Triplet> &train_vec, const int &begin_idx, const int &end_idx, const double &lambda,
               const double &mean_cnt, std::vector<double> &error,
               std::vector<double> &pred_out);

void reset(Mat &arr);

void Scale(Mat &res, const double &mult);

void Minus(Mat &res, const Mat & rhs);

void Add(Mat &res, const Mat & rhs);

double SqrSum(const std::vector<double> &arr);

extern std::vector<Triplet> train_vec, probe_vec;
extern std::vector<double> err_train, err_valid;

#endif //PLDA_PMF_MF_H
