/*
 * Scheduler
 * Solver
 *
 *
 *
 */


#include <iostream>
#include <vector>
#include <mpi.h>
#include <stddef.h>

#include "pmf.h"
#include "mf.h"

using namespace std;
using namespace pmf;

int my_rank, mpi_size;
MPI_Status status;

MPI_Datatype MPI_Triplet;

// Solver
void pmf::Solver::run(vector<vector<double> > &d_D, vector<vector<double> > &d_W) {
    int NN = (pairs_tr - 1) / mpi_size + 1; // number training triplets per batch

    vector<vector<double> > error = NewArray(NN, 1, 0), pred_out = NewArray(NN, 1, 0);

    vector<vector<double> > Ix_D = NewArray(NN, model.num_feat, 0), Ix_W = NewArray(NN, model.num_feat, 0); // gradient w.r.t. training sample

    double F = 0.0;

    for (int epoch = 0; epoch < maxepoch; epoch++) {
        // Random permute training data.
        std::random_shuffle(train_vec.begin(), train_vec.end());

        int begin_idx = (my_rank - 1) * NN, end_idx = min(my_rank * NN - 1, pairs_tr - 1), N = end_idx - begin_idx + 1;

        //%%%%%%%%%%%%%% Compute Predictions %%%%%%%%%%%%%%%%%
        F = CalcObj(model.D, model.W, train_vec, begin_idx, end_idx, model.lambda, mean_cnt, error, pred_out);

        //%%%%%%%%%%%%%% Compute Gradients %%%%%%%%%%%%%%%%%%%
        for (int i = begin_idx; i <= end_idx; i++) {
            int doc_id = train_vec[i].doc_id, word_id = train_vec[i].word_id;
            for (int j = 0; j < model.num_feat; j++) {
                double dd = model.D[doc_id][j], ww = model.W[word_id][j];
                Ix_D[i - begin_idx][j] = error[i - begin_idx][0] * 2 * ww + model.lambda * dd;
                Ix_W[i - begin_idx][j] = error[i - begin_idx][0] * 2 * dd + model.lambda * ww;
            }
        }

        reset(d_D);
        reset(d_W);
        for (int i = begin_idx; i <= end_idx; i++) {
            int doc_id = train_vec[i].doc_id, word_id = train_vec[i].word_id;
            for (int j = 0; j < model.num_feat; j++) {
                d_D[doc_id][j] += Ix_D[i - begin_idx][j];
                d_W[word_id][j] += Ix_W[i - begin_idx][j];
            }
        }

        //%%%%%%%%%%%%%% Compute Predictions after Parameter Updates %%%%%%%%%%%%%%%%%
        F = CalcObj(model.D, model.W, train_vec, begin_idx, end_idx, model.lambda, mean_cnt, error,
                    pred_out);
        err_train.push_back(sqrt(F / N));

        scheduler.sync();
    }
}

// GlobalScheduler
void pmf::GlobalScheduler::run() {
    for (int epoch = 1; epoch <= maxepoch; ++epoch) {
        sync();

        get_train_loss();
        get_probe_loss();

        printf("epoch %4i Training RMSE %6.4f  Test RMSE %6.4f  \n", epoch, err_train.back(), err_valid.back());
    }
}

void pmf::GlobalScheduler::update_weight() {
    cout << "Updating weights" << endl;

    //%%%% Update doc and word features %%%%%%%%%%%
    Scale(D_inc, D_inc, model.momentum);
    Scale(d_D, d_D, model.epsilon);
    Add(D_inc, D_inc, d_D);
    //D_inc = D_inc*momentum + d_D*(epsilon/N);
    Minus(model.D, model.D, D_inc);
    //D =  D - D_inc;

    Scale(W_inc, W_inc, model.momentum);
    Scale(d_W, d_W, model.epsilon);
    Add(W_inc, W_inc, d_W);
    //W_inc = W_inc*momentum + d_W*(epsilon/N);
    Minus(model.W, model.W, W_inc);
    //W =  W - W_inc;
}

void pmf::GlobalScheduler::sync() {
    cout << "Sync..." << endl;
    MPI_Barrier(MPI_COMM_WORLD);

    // Get d_D and d_W of each node
    _sync(d_D, model.num_d, model.num_feat);
    _sync(d_W, model.num_w, model.num_feat);
    cout << "d_D and d_W synced!" << endl;

    // Update D and W using d_D and d_W
    update_weight();

    // Send new D and W back
    cout << "Sync D and W" << endl;
    for (int i = 0; i < model.num_d; ++i) {
        MPI_Bcast(&model.D[i][0], model.num_feat, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    for (int i = 0; i < model.num_w; ++i) {
        MPI_Bcast(&model.W[i][0], model.num_feat, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    cout << "All synced!" << endl;
}

void pmf::GlobalScheduler::_sync(vector<vector<double> > &vec, int num_row, int num_col) {
    double *rows = new double[num_col * mpi_size];
    double *row = new double[num_col];
    for (int i = 0; i < num_row; ++i) {
        MPI_Gather(row, num_col, MPI_DOUBLE, rows, num_col, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        for (int j = 0; j < num_col; ++j) {
            double avg = 0.0;
            for (int k = 0; k < mpi_size; ++k) {
                avg += rows[k * num_col + j];
            }
            avg /= mpi_size;
            vec[i][j] = avg;
        }
    }
}

void pmf::GlobalScheduler::get_train_loss() {
    vector<vector<double> > error = NewArray(pairs_tr, 1, 0), pred_out = NewArray(pairs_tr, 1, 0);
    double F = CalcObj(model.D, model.W, train_vec, 0, pairs_tr - 1, model.lambda, mean_cnt, error, pred_out);
    err_train.push_back(sqrt(F / pairs_tr));
}

// Compute predictions on the validation set
void pmf::GlobalScheduler::get_probe_loss() {
    double mean_cnt = sum_cnt(train_vec) / pairs_tr;
    vector<vector<double> > test_error = NewArray(pairs_pr, 1, 0), test_pred_out = NewArray(pairs_pr, 1, 0);
    double F = CalcObj(model.D, model.W, probe_vec, 0, pairs_pr - 1, model.lambda, mean_cnt, test_error, test_pred_out);
    err_valid.push_back(sqrt(Sum(Sqr(test_error)) / pairs_pr));
}

// Local Scheduler
void pmf::LocalScheduler::run() {
    Solver solver(*this, train_block, model);

    for (int epoch = 1; epoch <= maxepoch; ++epoch) {
        // run Solver
        solver.run(d_D, d_W);

        // Sync parameters
        sync();
    }
}

void pmf::LocalScheduler::_sync(vector<vector<double> > vec, int num_row, int num_col) {
    double *rows = nullptr;
    for (int i = 0; i < num_row; ++i) {
        MPI_Gather(&vec[i][0], num_col, MPI_DOUBLE, rows, num_col, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}

void pmf::LocalScheduler::sync() {
    MPI_Barrier(MPI_COMM_WORLD);

    // Send d_D and d_W to master node
    _sync(d_D, model.num_d, model.num_feat);
    _sync(d_W, model.num_w, model.num_feat);

    // Get new D and W
    for (int i = 0; i < model.num_d; ++i) {
        MPI_Bcast(&model.D[i][0], model.num_feat, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    for (int i = 0; i < model.num_w; ++i) {
        MPI_Bcast(&model.W[i][0], model.num_feat, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}

vector<vector<double> > D, W;

pmf_model init_mode() {
    int num_d = 2020; // Number of docs
    int num_w = 7167; // Number of words
    int num_feat = 10; // Rank 10 decomposition
    D = NewArray(num_d, num_feat, RANDOM); // Doc feature vectors
    W = NewArray(num_w, num_feat, 0); // Word feature vecators

    pmf_model model(num_d, num_w, num_feat, D, W);
    model.epsilon = 5; // Learning rate
    model.lambda = 1; // Regularization parameter
    model.momentum = 0.9;
    return model;
}

void create_triplet_type() {
    const int num_elem = 3;
    int block_lengths[num_elem] = {1, 1, 1};
    MPI_Datatype types[num_elem] = {MPI_INT, MPI_INT, MPI_DOUBLE};
    MPI_Aint offsets[num_elem];
    offsets[0] = offsetof(TripletType, doc_id);
    offsets[1] = offsetof(TripletType, word_id);
    offsets[2] = offsetof(TripletType, cnt);
    MPI_Type_create_struct(num_elem, block_lengths, offsets, types, &MPI_Triplet);
    MPI_Type_commit(&MPI_Triplet);
}

/*
	main
*/
int main(int argc, char **argv) {
    string train_file = "";
    string probe_file = "";

    if (!ParseMainArgs(argc, argv, train_file, probe_file)) return 0;

    int maxepoch = 50;
    pmf_model model = init_mode();

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    create_triplet_type();

    int num_train;

    // Global Scheduler <-> Local Scheduler <-> Local Solver
    if (my_rank == 0) {
        // Global Scheduler
        int tmp_num_d;
        int cur_doc_id = 0;

        // load train and probe data
        cout << "Load Train and Probe Data." << endl;
        vector<Triplet> all_data_vec;
        load_data(train_file, all_data_vec, tmp_num_d, model.num_w, cur_doc_id); // Triplets: {doc_id, word_id, cnt}
        load_data(probe_file, all_data_vec, tmp_num_d, model.num_w, cur_doc_id);

        // shuffle data
        for (int i = 0; i < all_data_vec.size(); i++) {
            if (rand() % 100 < 90) train_vec.push_back(all_data_vec[i]);
            else probe_vec.push_back(all_data_vec[i]);
        }
        all_data_vec.clear();

        // transfer train_vec
        cout << "Transfering train_vec...";
        num_train = train_vec.size();
        MPI_Bcast(&num_train, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&train_vec[0], num_train, MPI_Triplet, 0, MPI_COMM_WORLD);
        cout << "...finished!" << endl;

        Block train_block(train_vec);
        Block probe_block(probe_vec);

        GlobalScheduler scheduler(model, train_block, probe_block, maxepoch, mpi_size);
        cout << "Running scheduler" << endl;
        scheduler.run();
    } else {
        // Local Scheduler

        // Sync train_vec
        MPI_Bcast(&num_train, 1, MPI_INT, 0, MPI_COMM_WORLD);
        train_vec.resize(num_train);
        MPI_Bcast(&train_vec[0], num_train, MPI_Triplet, 0, MPI_COMM_WORLD);

        Block train_block(train_vec);

        LocalScheduler scheduler(model, train_block, maxepoch, mpi_size);
        scheduler.run();
    }

    MPI_Finalize();

    return 0;
}
