// Concatenate after kernels.metal and optimized.metal. Each output retains the
// reference's ascending two-float accumulation; only ownership and reuse change.
#ifndef V3_QKV_FWD_M
#define V3_QKV_FWD_M 16
#define V3_QKV_FWD_N 32
#define V3_QKV_FWD_K 16
#endif
#ifndef V3_QKV_DX_M
#define V3_QKV_DX_M 16
#define V3_QKV_DX_N 32
#define V3_QKV_DX_K 16
#endif

template<uint TM,uint TN,uint TK,bool DX>
inline void v3_qkv_contract(device const float* x,device const float* weight,
                            device const float* bias,device float* y,constant Args& a,
                            uint2 group,uint tid,threadgroup float* left,
                            threadgroup float* right){
    constexpr uint NC=TN/16,THREADS=TM*8;
    static_assert((TM==16||TM==32)&&(TN==32||TN==64)&&(TK==16||TK==32),
                  "Unsupported protected QKV tile");
    uint inner=DX?a.n:a.k,cols=DX?a.k:a.n;
    uint row0=group.y*TM,col0=group.x*TN;
    uint localRow=2*(tid/16),localCol=NC*(tid%16);
    float2 accum[2*NC];
    for(uint z=0;z<2*NC;++z)accum[z]=float2(0);
    for(uint base=0;base<inner;base+=TK){
        for(uint i=tid;i<TM*TK;i+=THREADS){
            uint row=row0+i/TK,k=base+i%TK;
            left[i]=(row<a.m&&k<inner)?x[row*inner+k]:0;
        }
        // Forward weights are stored output-major; dX weights are input-major.
        // Both loads follow the contiguous physical dimension. Transpose the
        // forward tile into padded shared rows for the repeated channel reads.
        for(uint i=tid;i<TN*TK;i+=THREADS){
            uint kLocal=DX?i/TN:i%TK,colLocal=DX?i%TN:i/TK;
            uint k=base+kLocal,col=col0+colLocal;
            right[kLocal*(TN+1)+colLocal]=(k<inner&&col<cols)?(DX?weight[k*a.k+col]:weight[col*a.k+k]):0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint k=0;k<TK;++k){
            float l0=left[localRow*TK+k],l1=left[(localRow+1)*TK+k];
            for(uint c=0;c<NC;++c){
                float r=right[k*(TN+1)+localCol+c];
                accum[c]=ff_add(accum[c],ff_product(l0,r));
                accum[NC+c]=ff_add(accum[NC+c],ff_product(l1,r));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint r=0;r<2;++r)for(uint c=0;c<NC;++c){
        uint row=row0+localRow+r,col=col0+localCol+c;
        if(row<a.m&&col<cols){
            float2 value=accum[r*NC+c];
            if(!DX)value=ff_add(value,float2(bias[col],0));
            y[row*cols+col]=value.x+value.y;
        }
    }
}

kernel void v3_qkv_forward(device const float* x [[buffer(0)]],
                           device const float* weight [[buffer(1)]],
                           device const float* bias [[buffer(2)]],device float* y [[buffer(3)]],
                           constant Args& a [[buffer(7)]],uint2 group [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]){
    threadgroup float left[V3_QKV_FWD_M*V3_QKV_FWD_K],right[(V3_QKV_FWD_N+1)*V3_QKV_FWD_K];
    v3_qkv_contract<V3_QKV_FWD_M,V3_QKV_FWD_N,V3_QKV_FWD_K,false>(
        x,weight,bias,y,a,group,tid,left,right);
}

kernel void v3_qkv_dx(device const float* dy [[buffer(0)]],
                      device const float* weight [[buffer(1)]],device float* dx [[buffer(2)]],
                      constant Args& a [[buffer(7)]],uint2 group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]]){
    threadgroup float left[V3_QKV_DX_M*V3_QKV_DX_K],right[(V3_QKV_DX_N+1)*V3_QKV_DX_K];
    v3_qkv_contract<V3_QKV_DX_M,V3_QKV_DX_N,V3_QKV_DX_K,true>(
        dy,weight,weight,dx,a,group,tid,left,right);
}
