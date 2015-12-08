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

#include "mf.h" // everything

using namespace std;

/*
	main
*/
int main(int argc, char **argv) {
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

    for (int i = 0; i < all_data_vec.size(); i++) {
        if (rand() % 100 < 90) train_vec.push_back(all_data_vec[i]);
        else probe_vec.push_back(all_data_vec[i]);
    }
    all_data_vec.clear();

    int pairs_tr = train_vec.size(); // training data
    int pairs_pr = probe_vec.size(); // validation data

    double mean_cnt = sum_cnt(train_vec) / pairs_tr;

    int numbatches = 10; // Number of batches
    int NN = (pairs_tr - 1) / numbatches + 1; // number training triplets per batch

    vector<vector<double> > D = NewArray(num_d, num_feat, RANDOM); // Doc feature vectors
    vector<vector<double> > W = NewArray(num_w, num_feat, 0);// Word feature vecators
    vector<vector<double> > D_inc = NewArray(num_d, num_feat, 0), W_inc = NewArray(num_w, num_feat, 0);
    vector<vector<double> > error = NewArray(NN, 1, 0), pred_out = NewArray(NN, 1, 0), test_error = NewArray(pairs_pr, 1, 0), test_pred_out = NewArray(pairs_pr, 1, 0);

    vector<vector<double> > Ix_D = NewArray(NN, num_feat, 0), Ix_W = NewArray(NN, num_feat,
                                                                              0); // gradient w.r.t. training sample
    vector<vector<double> > d_D = NewArray(num_d, num_feat, 0), d_W = NewArray(num_w, num_feat, 0);

    double F = 0.0;

    for (; epoch < maxepoch; epoch++) {
        // Random permute training data.
        std::random_shuffle(train_vec.begin(), train_vec.end());

        for (int batch = 1; batch <= numbatches; batch++) {

            int begin_idx = (batch - 1) * NN, end_idx = min(batch * NN - 1, pairs_tr - 1), N = end_idx - begin_idx + 1;

            //%%%%%%%%%%%%%% Compute Predictions %%%%%%%%%%%%%%%%%
            F = CalcObj(D, W, train_vec, begin_idx, end_idx, lambda, mean_cnt, error, pred_out);

            //%%%%%%%%%%%%%% Compute Gradients %%%%%%%%%%%%%%%%%%%
            for (int i = begin_idx; i <= end_idx; i++) {
                int doc_id = train_vec[i].doc_id, word_id = train_vec[i].word_id;
                for (int j = 0; j < num_feat; j++) {
                    double dd = D[doc_id][j], ww = W[word_id][j];
                    Ix_D[i - begin_idx][j] = error[i - begin_idx][0] * 2 * ww + lambda * dd;
                    Ix_W[i - begin_idx][j] = error[i - begin_idx][0] * 2 * dd + lambda * ww;
                }
            }

            reset(d_D);
            reset(d_W);
            for (int i = begin_idx; i <= end_idx; i++) {
                int doc_id = train_vec[i].doc_id, word_id = train_vec[i].word_id;
                for (int j = 0; j < num_feat; j++) {
                    d_D[doc_id][j] += Ix_D[i - begin_idx][j];
                    d_W[word_id][j] += Ix_W[i - begin_idx][j];
                }
            }

            //%%%% Update doc and word features %%%%%%%%%%%
            Scale(D_inc, D_inc, momentum);
            Scale(d_D, d_D, epsilon / N);
            Add(D_inc, D_inc, d_D);
            //D_inc = D_inc*momentum + d_D*(epsilon/N);
            Minus(D, D, D_inc);
            //D =  D - D_inc;

            Scale(W_inc, W_inc, momentum);
            Scale(d_W, d_W, epsilon / N);
            Add(W_inc, W_inc, d_W);
            //W_inc = W_inc*momentum + d_W*(epsilon/N);
            Minus(W, W, W_inc);
            //W =  W - W_inc;

            //%%%%%%%%%%%%%% Compute Predictions after Parameter Updates %%%%%%%%%%%%%%%%%
            F = CalcObj(D, W, train_vec, begin_idx, end_idx, lambda, mean_cnt, error, pred_out);
            err_train.push_back(sqrt(F / N));

            //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            //%%% Compute predictions on the validation set %%%%%%%%%%%%%%%%%%%%%%
            F = CalcObj(D, W, probe_vec, 0, pairs_pr - 1, lambda, mean_cnt, test_error, test_pred_out);
            err_valid.push_back(sqrt(Sum(Sqr(test_error)) / pairs_pr));
            printf("epoch %4i batch %4i Training RMSE %6.4f  Test RMSE %6.4f  \n", epoch, batch, err_train.back(),
                   err_valid.back());
        }
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    }
}
