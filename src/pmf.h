//
// Created by Zheng Yan on 12/6/15.
//

#ifndef PLDA_PMF_PMF_H
#define PLDA_PMF_PMF_H

#include <string>
#include "mf.h"

namespace pmf {

    struct pmf_model {
        int num_d;
        int num_w;
        int num_feat;
        double epsilon;
        double lambda;
        double momentum;
        std::vector<std::vector<double> > &D; // Doc feature vectors
        std::vector<std::vector<double> > &W; // Word feature vecators

        pmf_model(int num_d, int num_w, int num_feat, std::vector<std::vector<double> > &D,
                  std::vector<std::vector<double> > &W) : num_d(num_d), num_w(num_w), num_feat(num_feat), D(D), W(W) { }
    };


    class Block {
    public:
        Block(std::vector<Triplet> &vec) : _vec(vec) { current = vec.begin(); };

        bool move_next() { return ++current != _vec.end(); };

        std::vector<Triplet>::iterator get_current() { return current; };

        size_t size() { return _vec.size(); }

    private:
        std::vector<Triplet> &_vec;
        std::vector<Triplet>::iterator current;
    };


    class Scheduler {
    public:
        Scheduler(pmf_model &model, Block &train_block, Block &probe_block, int maxepoch, int mpi_size) : model(model),
                                                                                                          train_block(
                                                                                                                  train_block),
                                                                                                          probe_block(
                                                                                                                  probe_block),
                                                                                                          mpi_size(
                                                                                                                  mpi_size),
                                                                                                          _is_terminated(
                                                                                                                  false),
                                                                                                          maxepoch(
                                                                                                                  maxepoch) { };

        Scheduler(const Scheduler &) = delete;

        Scheduler &operator=(const Scheduler &) = delete;

        bool is_terminated() { return _is_terminated; }

        void run();

        virtual void sync() { };

    protected:
        bool _is_terminated;
        pmf_model &model;
        std::vector<Block> blocks;
        Block &train_block, probe_block;
        int mpi_size;
        int maxepoch;
    };

    class LocalScheduler : Scheduler {
    public:
        LocalScheduler(pmf_model &model, Block &train_block, int maxepoch, int mpi_size) : Scheduler(model, train_block,
                                                                                                     train_block,
                                                                                                     maxepoch,
                                                                                                     mpi_size) {
            d_D = NewArray(model.num_d, model.num_feat, 0);
            d_W = NewArray(model.num_w, model.num_feat, 0);
        };

        void run();

        void sync();

    protected:
        std::vector<std::vector<double> > d_D, d_W;

        void _sync(std::vector<std::vector<double> > vec, int num_row, int num_col);
    };

    class GlobalScheduler : Scheduler {
    public:
        GlobalScheduler(pmf_model &model, Block &train_block, Block &probe_block, int maxepoch, int mpi_size)
                : Scheduler(model, train_block, probe_block, maxepoch, mpi_size),
                  epoch(1),
                  pairs_tr(train_block.size()),
                  pairs_pr(probe_block.size()) {
            D_inc = NewArray(model.num_d, model.num_feat, 0);
            W_inc = NewArray(model.num_w, model.num_feat, 0);
            d_D = NewArray(model.num_d, model.num_feat, 0);
            d_W = NewArray(model.num_w, model.num_feat, 0);
            mean_cnt = sum_cnt(train_vec) / pairs_tr;
        };

        void run();

        void sync();

        void update_weight();

        void get_train_loss();

        void get_probe_loss();

    protected:
        int epoch;
        size_t pairs_tr;
        size_t pairs_pr;
        std::vector<std::vector<double> > D_inc, W_inc;
        std::vector<std::vector<double> > d_D, d_W;
        double mean_cnt;

        void _sync(std::vector<std::vector<double> > &vec, int num_row, int num_col);
    };


    class Solver {
    public:
        Solver(Scheduler &scheduler, Block &block, pmf_model &model) : scheduler(scheduler), block(block), model(model) {
            pairs_tr = train_vec.size();
            mean_cnt = sum_cnt(train_vec) / pairs_tr;
        }

        void run(std::vector<std::vector<double> > &d_D, std::vector<std::vector<double> > &d_W);

        Solver(const Solver &) = delete;

        Solver &operator=(const Solver &) = delete;

    protected:
        Scheduler &scheduler;
        Block &block;
        int maxepoch;
        pmf_model &model;
        double mean_cnt;
        int pairs_tr;
    };
}

#endif //PLDA_PMF_PMF_H
