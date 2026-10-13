
#define JIT 1
#define JIT_cuda 1
#include <cmath>
#include <limits>
#include "var.h"
#include "ops/reduce_op.h"
#include "ops/op_register.h"
#include "executor.h"
#include <assert.h>
#include "fused_op.h"
#define op0_reduce 
#define op0_Tx float32
#define op0_Ty float32
#define op0_Tz float32
#define op0_OP mean
#define op0_DIM 4
#define op0_REDUCE 12
#define op0_index_t int32
#include "misc/cuda_atomic.h"
#include "misc/cuda_limits.h"
#include "helper_cuda.h"
using namespace jittor;
#define INLINE_FUNC inline static void 
__launch_bounds__(1024) __global__ void func_cb4bbd12c05af5bf_0(op0_index_t range0, op0_index_t range1, op0_Ty* __restrict__ op0_yp, int tn0) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int tn1 = 0;
    int tnum0 = 1<<(tn0-tn1);
    int tid0 = (thread_id>>tn1) & (tnum0-1);
    op0_index_t op0_yshape1 = range1;
    op0_index_t op0_yshape2 = 1;
    op0_index_t op0_yshape3 = 1;
    op0_index_t op0_ystride3 = 1;
    auto op0_ystride2 = op0_ystride3 * op0_yshape3;
    auto op0_ystride1 = op0_ystride2 * op0_yshape2;
    auto op0_ystride0 = op0_ystride1 * op0_yshape1;
    
    op0_index_t range01 = range1 * range0;
    for (op0_index_t id1 = tid0; id1 < range01; id1+=tnum0) {
        auto op0_yid = 0  + 0 * op0_ystride0 + id1 * op0_ystride1;
        op0_yp[op0_yid] = float32(0);
    }
}
__launch_bounds__(1024) __global__ void func_cb4bbd12c05af5bf_1(op0_index_t range0, op0_index_t range1, op0_index_t range2, op0_index_t range3, op0_Tx* __restrict__ op0_xp, op0_Ty* __restrict__ op0_yp, op0_Ty op0_rcount, int tn1, int tn0) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int tn2 = 0;
    int tnum0 = 1<<(tn0-tn1);
    int tid0 = (thread_id>>tn1) & (tnum0-1);
    int tnum1 = 1<<(tn1-tn2);
    int tid1 = (thread_id>>tn2) & (tnum1-1);
    op0_index_t op0_yshape1 = range1;
    op0_index_t op0_yshape2 = 1;
    op0_index_t op0_yshape3 = 1;
    op0_index_t op0_ystride3 = 1;
    auto op0_ystride2 = op0_ystride3 * op0_yshape3;
    auto op0_ystride1 = op0_ystride2 * op0_yshape2;
    auto op0_ystride0 = op0_ystride1 * op0_yshape1;
    op0_index_t op0_xstride3 = 1;
    auto op0_xstride2 = op0_xstride3 * range3;
    auto op0_xstride1 = op0_xstride2 * range2;
    auto op0_xstride0 = op0_xstride1 * range1;
    
    op0_index_t range01 = range1 * range0;
    op0_index_t range23 = range3 * range2;
    for (op0_index_t id1 = tid0; id1 < range01; id1+=tnum0) {
        {
            auto op0_yid = 0  + 0 * op0_ystride0 + id1 * op0_ystride1;
            float32 tmp0 = float32(0);
            for (op0_index_t id3 = tid1; id3 < range23; id3+=tnum1) {
                auto op0_xid = 0  + 0 * op0_xstride0 + id1 * op0_xstride1 + 0 * op0_xstride2 + id3 * op0_xstride3;
                tmp0=tmp0+((float32((op0_xp[op0_xid])))*(float32(op0_rcount)));
            }
            atomicAdd(&(op0_yp[op0_yid]),tmp0);
        }
    }
}
#pragma GCC diagnostic ignored "-Wunused-function"
inline static int get_thread_range_log(int& thread_num, int64 range) {
    int nbits = NanoVector::get_nbits(std::min((int64)thread_num, range)) - 2;
    thread_num >>= nbits;
    return nbits;
}
void jittor::FusedOp::jit_run() {
    auto op0_x = ((ReduceOp*)(ops[0]))->x;
    auto op0_y = ((ReduceOp*)(ops[0]))->y;
    op0_index_t range0 = op0_x->shape[0];
    op0_index_t range1 = op0_x->shape[1];
    op0_index_t range2 = op0_x->shape[2];
    op0_index_t range3 = op0_x->shape[3];
    auto* __restrict__ op0_xp = op0_x->ptr<op0_Tx>();
    auto* __restrict__ op0_yp = op0_y->ptr<op0_Ty>();
    op0_Ty op0_rcount = op0_y->num*1.0 / op0_x->num;
    
    {
        int thread_num = 2097152;
        int thread_num_left = thread_num;
        int tn0 = get_thread_range_log(thread_num_left, range1 * range0);
        tn0=std::max(tn0, 5);
        thread_num=1<<tn0;
        int p1 = std::max(thread_num/1024, 1);
        int p2 = std::min(thread_num, 1024);
        func_cb4bbd12c05af5bf_0<<<p1,p2>>>(range0,range1,op0_yp,tn0);
    }
    {
        int thread_num = ((( range1 * range0))<=256)?std::max(524288/2,65536):524288;
        int thread_num_left = thread_num;
        int tn0 = get_thread_range_log(thread_num_left, range1 * range0);
        int tn1 = get_thread_range_log(thread_num_left, std::min<int64>( range3 * range2, std::max<int64>( range3 * range2/4,32)));
        if (tn1<5) {
            int _ = std::min(5-tn1,tn0);
            tn1+=_;
            tn0-=_;
        }
        tn0=tn0+tn1;
        tn0=std::max(tn0, 5);
        thread_num=1<<tn0;
        int p1 = std::max(thread_num/1024, 1);
        int p2 = std::min(thread_num, 1024);
        func_cb4bbd12c05af5bf_1<<<p1,p2>>>(range0,range1,range2,range3,op0_xp,op0_yp,op0_rcount,tn1,tn0);
    }
}
