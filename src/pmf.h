//
// Created by Zheng Yan on 12/6/15.
//

#ifndef PLDA_PMF_PMF_H
#define PLDA_PMF_PMF_H

#include <string>
#include <mpi.h>

#include "mf.h"

extern MPI_Status status;

namespace pmf {

    struct pmf_model {
        int num_d;
        int num_w;
        int num_feat;
        double epsilon;
        double lambda;
        double momentum;

        Mat &D; // Doc feature vectors
        Mat &W; // Word feature vecators

        pmf_model() : num_d(0), num_w(0), num_feat(0), D(*(new Mat(0,0,0))), W(*(new Mat(0,0,0))) { }

        pmf_model(int num_d, int num_w, int num_feat, Mat &D, Mat &W)
                : num_d(num_d), num_w(num_w), num_feat(num_feat), D(D), W(W) { }
    };


    class Block {
    public:
        Block(std::vector<Triplet> &vec) : _vec(vec) { current = vec.begin(); };

        Block() : _vec(*(new std::vector<Triplet>())) { current = _vec.begin(); };

        bool move_next() { return ++current != _vec.end(); };

        std::vector<Triplet>::iterator get_current() { return current; };

        size_t size() { return _vec.size(); }

        void append(Triplet triplet) { _vec.push_back(triplet); };

    private:
        std::vector<Triplet> &_vec;
        std::vector<Triplet>::iterator current;
    };


    class Scheduler {
    public:
        Scheduler(pmf_model &model, Block &train_block, Block &probe_block, int maxepoch, int mpi_size)
                : model(model), train_block(train_block), probe_block(probe_block), mpi_size(mpi_size),
                  maxepoch(maxepoch), d_D(model.num_d, model.num_feat, 0), d_W(model.num_w, model.num_feat, 0),
                  D_inc(model.num_d, model.num_feat, 0), W_inc(model.num_w, model.num_feat, 0) {
            pairs_tr = train_block.size();
            pairs_pr = probe_block.size();
            mean_cnt = sum_cnt(train_vec) / pairs_tr;
        };

        Scheduler(const Scheduler &) = delete;

        Scheduler &operator=(const Scheduler &) = delete;

        void run();

        virtual void sync() { };

        void update_weight();

        void get_train_loss();

        void get_probe_loss();

    protected:
        Mat d_D, d_W;
        Mat D_inc, W_inc;

        double mean_cnt;
        pmf_model &model;
        std::vector<Block> blocks;
        Block &train_block, probe_block;
        int mpi_size;
        int maxepoch;
        size_t pairs_tr;
        size_t pairs_pr;

        virtual void _sync(Mat &mat, int num_row, int num_col) {};
    };

    class LocalScheduler : Scheduler {
    public:
        LocalScheduler(pmf_model &model, Block &train_block, int maxepoch, int mpi_size)
                : Scheduler(model, train_block, train_block, maxepoch, mpi_size) { };

        void run();

        void sync();

    protected:
        void _sync(Mat &mat, int num_row, int num_col);
    };

    class GlobalScheduler : Scheduler {
    public:
        GlobalScheduler(pmf_model &model, Block &train_block, Block &probe_block, int maxepoch, int mpi_size)
                : Scheduler(model, train_block, probe_block, maxepoch, mpi_size), epoch(1) { };

        void run();

        void sync();

    protected:
        int epoch;

        void _sync(Mat &mat, int num_row, int num_col);
    };


    class Solver {
    public:
        Solver(Scheduler &scheduler, Block &block, pmf_model &model) : scheduler(scheduler), block(block),
                                                                       model(model) {
            pairs_tr = train_vec.size();
            mean_cnt = sum_cnt(train_vec) / pairs_tr;
        }

        void run(Mat &d_D, Mat &d_W);

        Solver(const Solver &) = delete;

        Solver &operator=(const Solver &) = delete;

    protected:
        Scheduler &scheduler;
        Block &block;
        pmf_model &model;
        double mean_cnt;
        int pairs_tr;
    };

}

#endif //PLDA_PMF_PMF_H
