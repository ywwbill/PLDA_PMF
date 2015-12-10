//
// Created by Zheng Yan on 12/9/15.
//

#include <iostream>
#include <mpi.h>

#include "pmf.h"
#include "pmf_block.h"

using namespace std;

void pmf::BlockSolver::run(Block &block, Mat &d_D, Mat &d_W, int curt_block) {
    int NN = block.size(); // number training triplets per batch
    vector<double> error(NN, 0), pred_out(NN, 0);
    Mat Ix_D(NN, model.num_feat, 0), Ix_W(NN, model.num_feat, 0); // gradient w.r.t. training sample
    double F = 0.0;
    int begin_idx = 0, end_idx = block.size() - 1;

    //%%%%%%%%%%%%%% Compute Predictions %%%%%%%%%%%%%%%%%
    F = CalcObj(model.D, model.W, train_vec, begin_idx, end_idx, model.lambda, mean_cnt, error, pred_out);

    //%%%%%%%%%%%%%% Compute Gradients %%%%%%%%%%%%%%%%%%%
    for (int i = begin_idx; i <= end_idx; i++) {
        int doc_id = train_vec[i].doc_id, word_id = train_vec[i].word_id;
        for (int j = 0; j < model.num_feat; j++) {
            double dd = model.D.get(doc_id, j), ww = model.W.get(word_id, j);
            Ix_D.set(i - begin_idx, j, error[i - begin_idx] * 2 * ww + model.lambda * dd);
            Ix_W.set(i - begin_idx, j, error[i - begin_idx] * 2 * dd + model.lambda * ww);
        }
    }

    reset(d_D);
    reset(d_W);
    for (int i = begin_idx; i <= end_idx; i++) {
        int doc_id = train_vec[i].doc_id, word_id = train_vec[i].word_id;
        for (int j = 0; j < model.num_feat; j++) {
            d_D.set(doc_id, j, d_D.get(doc_id,j)+Ix_D.get(i - begin_idx,j));
            d_W.set(word_id, j, d_W.get(word_id, j) + Ix_W.get(i - begin_idx,j));
        }
    }
}

void pmf::BlockGlobalScheduler::run() {
    for (int epoch = 1; epoch <= maxepoch; ++epoch) {
        sync(epoch);

        get_train_loss();
        get_probe_loss();

        printf("epoch %4i Training RMSE %6.4f  Test RMSE %6.4f  \n", epoch, err_train.back(), err_valid.back());
    }
}

void pmf::BlockGlobalScheduler::update_weight() {
    // only update W, D is updated locally
    Scale(W_inc, model.momentum);
    Scale(d_W, model.epsilon);
    Add(W_inc, d_W);
    //W_inc = W_inc*momentum + d_W*(epsilon/N);
    Minus(model.W, W_inc);
    //W =  W - W_inc;
}

void pmf::BlockGlobalScheduler::sync(int epoch) {
    cout << "Sync..." << endl;
    MPI_Barrier(MPI_COMM_WORLD);

    // Get d_D and d_W of each node
    int doc_length = (model.num_d - 1) / (mpi_size - 1) + 1;
    int word_length = (model.num_w - 1) / (mpi_size - 1) + 1;

    for (int i = 1; i <= mpi_size; ++i) {
        int word_begin_idx = ((i - 1 + epoch) % mpi_size) * (word_length / (mpi_size - 1));
        MPI_Recv(d_W.arr + word_begin_idx * model.num_feat, word_length * model.num_feat, MPI_DOUBLE, i, 99, MPI_COMM_WORLD, &status);
    }
    cout << "d_D and d_W synced!" << endl;

    // Update W using d_W
    update_weight();

    // Send new part of W back
    for (int i = 1; i <= mpi_size; ++i) {
        int word_begin_idx = ((i + epoch) % mpi_size) * (word_length / (mpi_size - 1));
        MPI_Send(model.W.arr + word_begin_idx * model.num_feat, word_length * model.num_feat, MPI_DOUBLE, i, 99, MPI_COMM_WORLD);
    }
    cout << "All synced!" << endl;
}

void pmf::BlockLocalScheduler::sync() {
    int doc_length = (model.num_d - 1) / (mpi_size - 1) + 1;
    int word_length = (model.num_w - 1) / (mpi_size - 1) + 1;
    int word_begin_idx = curt_block * (word_length / (mpi_size - 1));
    int word_begin_idx_next = ((curt_block + 1) % mpi_size ) * (word_length / (mpi_size - 1));

    MPI_Send(d_W.arr + word_begin_idx * model.num_feat, word_length * model.num_feat, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD);
    MPI_Recv(model.W.arr + word_begin_idx_next * model.num_feat, word_length * model.num_feat, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD, &status);
}

void pmf::BlockLocalScheduler::run() {
    BlockSolver solver(*this, train_block, model, doc_begin_idx, doc_end_idx);

    for (int epoch = 1; epoch <= maxepoch; ++epoch) {
        // run Solver
        solver.run(blocks[curt_block], d_D, d_W, curt_block);

        // Sync parameters
        sync();

        // move to next block
        curt_block = (curt_block + 1) % mpi_size;
    }
}

void pmf::BlockLocalScheduler::partition() {
    blocks.resize(mpi_size - 1);
    for (int i = 0; i < train_vec.size(); ++i) {
        int doc_id = train_vec[i].doc_id, word_id = train_vec[i].word_id;
        if (doc_id < doc_begin_idx || doc_id > doc_end_idx)
            continue;
        blocks[ word_id / (model.num_w / (mpi_size - 1)) ].append(train_vec[i]);
    }
}
