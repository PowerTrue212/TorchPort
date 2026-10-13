#define JIT 1
#define JIT_cuda 1






#include <cmath>
#include "var.h"
#include "ops/code_op.h"
#include "ops/op_register.h"
#include "misc/cuda_flags.h"
#define code 
#define IN_SIZE 1
#define in0_dim 4
#define in0_type float32
#define OUT_SIZE 1
#define out0_dim 2
#define out0_type float32
#define index_t int32

#define __inline_static__ inline static



#pragma GCC diagnostic ignored "-Wunused-variable"


    #define in0_stride3 1


    #define out0_stride1 1


#define ARGS_DEF   \
  \
    in0_type* __restrict__ in0_p, \
     index_t in0_shape0, index_t in0_shape1, index_t in0_shape2, index_t in0_shape3, \
 \
  \
    out0_type* __restrict__ out0_p, \
     index_t out0_shape0, index_t out0_shape1, \
 \
int __tmp \


#define ARGS   \
  \
    in0_p, \
     in0_shape0, in0_shape1, in0_shape2, in0_shape3, \
 \
  \
    out0_p, \
     out0_shape0, out0_shape1, \
 \
0 \


#define PRECALC  \
 \
     auto in0_stride2 = in0_stride3 * in0_shape3; auto in0_stride1 = in0_stride2 * in0_shape2; auto in0_stride0 = in0_stride1 * in0_shape1; \
 \
 \
     auto out0_stride0 = out0_stride1 * out0_shape1; \
 \



#define out out0
#define out_p out0_p
#define out_dim 2
#define out_type float32
#define out_stride0 out0_stride0
#define out_shape0 out0_shape0
#define out_stride1 1
#define out_shape1 out0_shape1

#undef out


namespace jittor {

__global__ static void fixed_tree_spatial_sum( 
 
    in0_type* __restrict__ in0_p,
     index_t in0_shape0, index_t in0_shape1, index_t in0_shape2, index_t in0_shape3,

 
    out0_type* __restrict__ out0_p,
     index_t out0_shape0, index_t out0_shape1,

int __tmp
) {
    

     auto in0_stride2 = in0_stride3 * in0_shape3; auto in0_stride1 = in0_stride2 * in0_shape2; auto in0_stride0 = in0_stride1 * in0_shape1;


     auto out0_stride0 = out0_stride1 * out0_shape1;


    const int output_id = blockIdx.x;
    const int output_count = out_shape0 * out_shape1;
    if (output_id >= output_count) return;
    const int n = output_id / in0_shape1;
    const int c = output_id % in0_shape1;
    const int spatial = in0_shape2 * in0_shape3;
    __shared__ float partial[256];
    float local = 0.0f;
    for (int k = threadIdx.x; k < spatial; k += blockDim.x) {
        local += in0_p[(n)*in0_stride0+( c)*in0_stride1+( k / in0_shape3)*in0_stride2+( k % in0_shape3)*in0_stride3];
    }
    partial[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out_p[output_id] = partial[0];
}

}

#define out out0

namespace jittor {

void CodeOp::jit_run() {
    
    
        auto in0 = _inputs[0];
        auto* __restrict__ in0_p = _inputs[0]->ptr<in0_type>();
         index_t in0_shape0 = _inputs[0]->shape[0]; index_t in0_shape1 = _inputs[0]->shape[1]; index_t in0_shape2 = _inputs[0]->shape[2]; index_t in0_shape3 = _inputs[0]->shape[3];
    
    
    
        auto out0 = _outputs[0];
        auto* __restrict__ out0_p = _outputs[0]->ptr<out0_type>();
         index_t out0_shape0 = _outputs[0]->shape[0]; index_t out0_shape1 = _outputs[0]->shape[1];
    

    

     auto in0_stride2 = in0_stride3 * in0_shape3; auto in0_stride1 = in0_stride2 * in0_shape2; auto in0_stride0 = in0_stride1 * in0_shape1;


     auto out0_stride0 = out0_stride1 * out0_shape1;



    fixed_tree_spatial_sum<<<out_shape0 * out_shape1, 256>>>( 
 
    in0_p,
     in0_shape0, in0_shape1, in0_shape2, in0_shape3,

 
    out0_p,
     out0_shape0, out0_shape1,

0
);

}

} 



