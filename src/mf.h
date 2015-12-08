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

bool ParseMainArgs(int argc, char **argv, std::string &train_file, std::string &probe_file);

void load_data(std::string data_file, std::vector<Triplet> &data_vec, int &num_d, int &num_w, int &cur_doc_id);

double sum_cnt(std::vector<Triplet> &data_vec);

std::vector<std::vector<double> > NewArray(int n, int m, int default_val);

double CalcObj(const std::vector<std::vector<double> > &D, const std::vector<std::vector<double> > &W,
               const std::vector<Triplet> &train_vec, const int &begin_idx, const int &end_idx, const double &lambda,
               const double &mean_cnt, std::vector<std::vector<double> > &error,
               std::vector<std::vector<double> > &pred_out);

void reset(std::vector<std::vector<double> > &arr);

void Scale(std::vector<std::vector<double> > &res, const std::vector<std::vector<double> > lhs, const double &mult);

void Minus(std::vector<std::vector<double> > &res, const std::vector<std::vector<double> > lhs,
           const std::vector<std::vector<double> > rhs);

void Add(std::vector<std::vector<double> > &res, const std::vector<std::vector<double> > lhs,
         const std::vector<std::vector<double> > rhs);

double Sum(const std::vector<std::vector<double> > &arr);

std::vector<std::vector<double> > Sqr(const std::vector<std::vector<double> > &arr);

extern std::vector<Triplet> train_vec, probe_vec;
extern std::vector<double> err_train, err_valid;

#endif //PLDA_PMF_MF_H
