//
// Created by Zheng Yan on 12/9/15.
//

#ifndef PLDA_PMF_PMF_BLOCK_H
#define PLDA_PMF_PMF_BLOCK_H

#include "pmf.h"

namespace pmf {

    // TODO: implement block partition scheduler
    class BlockGlobalScheduler : Scheduler {
    public:
        BlockGlobalScheduler(pmf_model &model, Block &train_block, Block &probe_block, int maxepoch)
                : Scheduler(model, train_block, probe_block, maxepoch) { };

        void run();

        void sync(int epoch);

        void update_weight();

    protected:
        void _sync();
    };

    class BlockLocalScheduler : Scheduler {
    public:
        BlockLocalScheduler(pmf_model &model, Block &train_block, int maxepoch)
                : Scheduler(model, train_block, train_block, maxepoch) {
            int NN = (model.num_d - 1) / (mpi_size - 1);
            doc_begin_idx = (my_rank - 1) * NN;
            doc_end_idx = std::min(my_rank * NN - 1, model.num_d - 1);
            curt_block = my_rank - 1;
        };

        void run();

        void sync();

        void partition();

        void update_weight();

    protected:
        int doc_begin_idx;
        int doc_end_idx;
        int curt_block;
        std::vector <Block> blocks;
    };

    class BlockSolver : Solver {
    public:
        BlockSolver(Scheduler &scheduler, Block &block, pmf_model &model, int doc_begin_idx, int doc_end_idx)
                : Solver(scheduler, block, model), doc_begin_idx(doc_begin_idx), doc_end_idx(doc_end_idx) {};

        void run(Block &block, Mat &d_D, Mat &d_W, int curt_block);

    protected:
        int doc_begin_idx;
        int doc_end_idx;
    };

    // TODO: implement lock free scheduler
    class LockFreeLocalScheduler : Scheduler {
        LockFreeLocalScheduler(pmf_model &model, Block &train_block, int maxepoch)
                : Scheduler(model, train_block, train_block, maxepoch) { };
    };

};

#endif //PLDA_PMF_PMF_BLOCK_H_H
