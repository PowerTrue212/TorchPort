
#define JIT 1
#define JIT_cuda 1
#include <cmath>
#include <algorithm>
#include "var.h"
#include "ops/broadcast_to_op.h"
#include "ops/op_register.h"
#include <cmath>
#include "var.h"
#include "ops/reindex_op.h"
#include "ops/op_register.h"
#include <cmath>
#include "var.h"
#include "ops/binary_op.h"
#include "ops/broadcast_to_op.h"
#include "ops/op_register.h"
#include <cmath>
#include <limits>
#include "var.h"
#include "ops/reduce_op.h"
#include "ops/op_register.h"
#include "executor.h"
#include <assert.h>
#include "fused_op.h"
#define op0_broadcast_to 
#define op0_Tx float32
#define op0_DIM 7
#define op0_BCAST 25
#define op0_index_t int32
#define op1_reindex 
#define op1_Tx float32
#define op1_XDIM 4
#define op1_YDIM 7
#define op1_OVERFLOW 0x0p0
#define op1_INDEX0 i0
#define op1_INDEX1 i2
#define op1_INDEX2 i3*1-1+i5*1
#define op1_INDEX3 i4*1-1+i6*1
#define op1_OSIZE 0
#define op1_ESIZE 0
#define op1_index_t int32
#define op2_binary 
#define op2_Tx float32
#define op2_Ty float32
#define op2_Tz float32
#define op2_OP multiply
#define op2_index_t int32
#define op3_reduce 
#define op3_Tx float32
#define op3_Ty float32
#define op3_Tz float32
#define op3_OP add
#define op3_DIM 7
#define op3_REDUCE 100
#define op3_index_t int32
#include "misc/cuda_atomic.h"
#include "misc/cuda_limits.h"
#include "helper_cuda.h"
using namespace jittor;
#define INLINE_FUNC inline static void 
__launch_bounds__(1024) __global__ void func_f0a5bd3515b87c91_0(op3_index_t range0, op3_index_t range1, op3_index_t range3, op3_index_t range4, op3_Ty* __restrict__ op3_yp, int tn0) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int tn1 = 0;
    int tnum0 = 1<<(tn0-tn1);
    int tid0 = (thread_id>>tn1) & (tnum0-1);
    op3_index_t op3_yshape1 = range1;
    op3_index_t op3_yshape2 = 1;
    op3_index_t op3_yshape3 = range3;
    op3_index_t op3_yshape4 = range4;
    op3_index_t op3_yshape5 = 1;
    op3_index_t op3_yshape6 = 1;
    op3_index_t op3_ystride6 = 1;
    auto op3_ystride5 = op3_ystride6 * op3_yshape6;
    auto op3_ystride4 = op3_ystride5 * op3_yshape5;
    auto op3_ystride3 = op3_ystride4 * op3_yshape4;
    auto op3_ystride2 = op3_ystride3 * op3_yshape3;
    auto op3_ystride1 = op3_ystride2 * op3_yshape2;
    auto op3_ystride0 = op3_ystride1 * op3_yshape1;
    
    op3_index_t range0134 = range1*range3*range4 * range0;
    for (op3_index_t id4 = tid0; id4 < range0134; id4+=tnum0) {
        auto op3_yid = 0  + 0 * op3_ystride0 + 0 * op3_ystride1 + 0 * op3_ystride3 + id4 * op3_ystride4;
        op3_yp[op3_yid] = float32(0);
    }
}
__launch_bounds__(1024) __global__ void func_f0a5bd3515b87c91_1(op3_index_t range0, op3_index_t range1, op3_index_t range2, op3_index_t range3, op3_index_t range4, op3_index_t range5, op3_index_t range6, op0_Tx* __restrict__ op0_xp, op1_Tx* __restrict__ op1_xp, op1_index_t op1_xshape0, op1_index_t op1_xshape1, op1_index_t op1_xshape2, op1_index_t op1_xshape3, op3_Ty* __restrict__ op3_yp, int tn3, int tn2, int tn1, int tn0) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int tn4 = 0;
    int tnum0 = 1<<(tn0-tn1);
    int tid0 = (thread_id>>tn1) & (tnum0-1);
    int tnum1 = 1<<(tn1-tn2);
    int tid1 = (thread_id>>tn2) & (tnum1-1);
    int tnum2 = 1<<(tn2-tn3);
    int tid2 = (thread_id>>tn3) & (tnum2-1);
    int tnum3 = 1<<(tn3-tn4);
    int tid3 = (thread_id>>tn4) & (tnum3-1);
    op0_index_t op0_xstride6 = 1;
    auto op0_xstride5 = op0_xstride6 * range6;
    auto op0_xstride4 = op0_xstride5 * range5;
    auto op0_xstride3 = op0_xstride4 * 1;
    auto op0_xstride2 = op0_xstride3 * 1;
    auto op0_xstride1 = op0_xstride2 * range2;
    auto op0_xstride0 = op0_xstride1 * range1;
    op1_index_t op1_xstride3 = 1;
    auto op1_xstride2 = op1_xstride3 * op1_xshape3;
    auto op1_xstride1 = op1_xstride2 * op1_xshape2;
    auto op1_xstride0 = op1_xstride1 * op1_xshape1;
    op3_index_t op3_yshape1 = range1;
    op3_index_t op3_yshape2 = 1;
    op3_index_t op3_yshape3 = range3;
    op3_index_t op3_yshape4 = range4;
    op3_index_t op3_yshape5 = 1;
    op3_index_t op3_yshape6 = 1;
    op3_index_t op3_ystride6 = 1;
    auto op3_ystride5 = op3_ystride6 * op3_yshape6;
    auto op3_ystride4 = op3_ystride5 * op3_yshape5;
    auto op3_ystride3 = op3_ystride4 * op3_yshape4;
    auto op3_ystride2 = op3_ystride3 * op3_yshape3;
    auto op3_ystride1 = op3_ystride2 * op3_yshape2;
    auto op3_ystride0 = op3_ystride1 * op3_yshape1;
    
    for (op3_index_t id0 = tid0; id0 < range0; id0+=tnum0) {
        for (op3_index_t id1 = tid1; id1 < range1; id1+=tnum1) {
            for (op3_index_t id3 = tid3; id3 < range3; id3+=tnum3) {
                for (op3_index_t id4 = 0; id4 < range4; id4++) {
                    {
                        auto op3_yid = 0  + id0 * op3_ystride0 + id1 * op3_ystride1 + id3 * op3_ystride3 + id4 * op3_ystride4;
                        float32 tmp0 = float32(0);
                        for (op3_index_t id2 = tid2; id2 < range2; id2+=tnum2) {
                            for (op3_index_t id5 = 0; id5 < range5; id5++) {
                                for (op3_index_t id6 = 0; id6 < range6; id6++) {
                                    auto op0_xid = + 0 * op0_xstride0 + id1 * op0_xstride1 + id2 * op0_xstride2 + 0 * op0_xstride3 + 0 * op0_xstride4 + id5 * op0_xstride5 + id6 * op0_xstride6;
                                    auto op0_zd = op0_xp[op0_xid];
                                    op1_index_t op1_xid0 = id0;
                                    op1_index_t op1_xid1 = id2;
                                    op1_index_t op1_xid2 = id3*1-1+id5*1;
                                    op1_index_t op1_xid3 = id4*1-1+id6*1;
                                    auto op1_xid = + op1_xid0 * op1_xstride0 + op1_xid1 * op1_xstride1 + op1_xid2 * op1_xstride2 + op1_xid3 * op1_xstride3;
                                    bool op1_check_overflow = 0  || op1_xid0<0 || op1_xid0>=op1_xshape0 || op1_xid1<0 || op1_xid1>=op1_xshape1 || op1_xid2<0 || op1_xid2>=op1_xshape2 || op1_xid3<0 || op1_xid3>=op1_xshape3;
                                    auto op1_yd = op1_check_overflow ? op1_Tx(0x0p0) : op1_xp[op1_xid];
                                    auto op2_zd = ((op1_yd       )*(op0_zd       ));
                                    tmp0=tmp0+op2_zd;
                                }
                            }
                        }
                        atomicAdd(&(op3_yp[op3_yid]),tmp0);
                    }
                }
            }
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
    auto op0_x = ((BroadcastToOp*)(ops[0]))->x;
    auto op1_x = ((ReindexOp*)(ops[1]))->x;
    auto op2_z = ((BinaryOp*)(ops[2]))->z;
    auto op3_y = ((ReduceOp*)(ops[3]))->y;
    op3_index_t range0 = op2_z->shape[0];
    op3_index_t range1 = op2_z->shape[1];
    op3_index_t range2 = op2_z->shape[2];
    op3_index_t range3 = op2_z->shape[3];
    op3_index_t range4 = op2_z->shape[4];
    op3_index_t range5 = op2_z->shape[5];
    op3_index_t range6 = op2_z->shape[6];
    auto* __restrict__ op0_xp = op0_x->ptr<op0_Tx>();
    auto* __restrict__ op1_xp = op1_x->ptr<op1_Tx>();
    op1_index_t op1_xshape0 = op1_x->shape[0];
    op1_index_t op1_xshape1 = op1_x->shape[1];
    op1_index_t op1_xshape2 = op1_x->shape[2];
    op1_index_t op1_xshape3 = op1_x->shape[3];
    auto* __restrict__ op3_yp = op3_y->ptr<op3_Ty>();
    
    {
        int thread_num = 2097152;
        int thread_num_left = thread_num;
        int tn0 = get_thread_range_log(thread_num_left, range1*range3*range4 * range0);
        tn0=std::max(tn0, 5);
        thread_num=1<<tn0;
        int p1 = std::max(thread_num/1024, 1);
        int p2 = std::min(thread_num, 1024);
        func_f0a5bd3515b87c91_0<<<p1,p2>>>(range0,range1,range3,range4,op3_yp,tn0);
    }
    {
        int thread_num = 2097152;
        int thread_num_left = thread_num;
        int tn3 = get_thread_range_log(thread_num_left, range3);
        int tn1 = get_thread_range_log(thread_num_left, range1);
        int tn0 = get_thread_range_log(thread_num_left, range0);
        int tn2 = get_thread_range_log(thread_num_left, range2);
        tn2=tn2+tn3;
        tn1=tn1+tn2;
        tn0=tn0+tn1;
        tn0=std::max(tn0, 5);
        thread_num=1<<tn0;
        int p1 = std::max(thread_num/1024, 1);
        int p2 = std::min(thread_num, 1024);
        func_f0a5bd3515b87c91_1<<<p1,p2>>>(range0,range1,range2,range3,range4,range5,range6,op0_xp,op1_xp,op1_xshape0,op1_xshape1,op1_xshape2,op1_xshape3,op3_yp,tn3,tn2,tn1,tn0);
    }
}
