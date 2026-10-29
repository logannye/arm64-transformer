// Concatenate after kernels.metal. Reference kernels remain available unchanged.
#include <metal_simdgroup_matrix>

#ifndef V2_MAT_M
#define V2_MAT_M 16
#endif
#ifndef V2_MAT_N
#define V2_MAT_N 32
#endif
#ifndef V2_MAT_K
#define V2_MAT_K 16
#endif
#ifndef V2_MAT_SG
#define V2_MAT_SG 2
#endif
#ifndef V2_FWD_M
#define V2_FWD_M V2_MAT_M
#define V2_FWD_N V2_MAT_N
#define V2_FWD_K V2_MAT_K
#define V2_FWD_SG V2_MAT_SG
#endif
#ifndef V2_DX_M
#define V2_DX_M V2_MAT_M
#define V2_DX_N V2_MAT_N
#define V2_DX_K V2_MAT_K
#define V2_DX_SG V2_MAT_SG
#endif
#ifndef V2_DW_M
#define V2_DW_M V2_MAT_M
#define V2_DW_N V2_MAT_N
#define V2_DW_K V2_MAT_K
#define V2_DW_SG V2_MAT_SG
#endif
#ifndef V2_NORM_SG
#define V2_NORM_SG 1
#endif
#ifndef V2_NORM_VEC
#define V2_NORM_VEC 1
#endif

// MODE 0: X W^T; MODE 1: dY W; MODE 2: dY^T X.
// Every SIMD group owns complete 8x8 output tiles. Cooperative guarded loads
// make the same kernels valid for short final batches and non-tile row counts.
template<uint TM,uint TN,uint TK,uint NSG,uint MODE,bool POST>
inline void v2_dense(device const float* x,device const float* w,
                     device const float* bias,device float* y,constant Args& a,
                     uint2 group,uint tid,uint sg,threadgroup float* left,
                     threadgroup float* right,threadgroup float* result,
                     device const float* residual,device float* fused){
    constexpr uint NT=(TM/8)*(TN/8)/NSG;
    simdgroup_float8x8 accum[NT];
    for(uint z=0;z<NT;++z)accum[z]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
    uint rows=MODE==2?a.n:a.m,cols=MODE==0?a.n:a.k;
    uint inner=MODE==0?a.k:(MODE==1?a.n:a.m);
    uint row0=group.y*TM,col0=group.x*TN;
    for(uint base=0;base<inner;base+=TK){
        for(uint i=tid;i<TM*TK;i+=NSG*32){
            uint r=row0+i/TK,k=base+i%TK;float value=0;
            if(r<rows&&k<inner)value=MODE==2?w[k*a.n+r]:x[r*(MODE==0?a.k:a.n)+k];
            left[i]=value;
        }
        for(uint i=tid;i<TK*TN;i+=NSG*32){
            uint k=base+i/TN,c=col0+i%TN;float value=0;
            if(k<inner&&c<cols)value=MODE==0?w[c*a.k+k]:(MODE==1?w[k*a.k+c]:x[k*a.k+c]);
            right[i]=value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint k=0;k<TK;k+=8){
            for(uint z=0;z<NT;++z){
                uint tile=sg+z*NSG,tr=(tile/(TN/8))*8,tc=(tile%(TN/8))*8;
                simdgroup_float8x8 l,r;
                simdgroup_load(l,left+tr*TK+k,TK);
                simdgroup_load(r,right+k*TN+tc,TN);
                simdgroup_multiply_accumulate(accum[z],l,r,accum[z]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint z=0;z<NT;++z){
        uint tile=sg+z*NSG,tr=(tile/(TN/8))*8,tc=(tile%(TN/8))*8;
        simdgroup_store(accum[z],result+tr*TN+tc,TN);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<TM*TN;i+=NSG*32){
        uint r=row0+i/TN,c=col0+i%TN;
        if(r<rows&&c<cols){
            float value=result[i]+(MODE==0?bias[c]:0.0f);y[r*cols+c]=value;
            if(POST)fused[r*cols+c]=a.step==1?max(value,0.0f):value+residual[r*cols+c];
        }
    }
}

kernel void v2_linear_forward(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],
                              device const float* bias [[buffer(2)]],device float* y [[buffer(3)]],
                              constant Args& a [[buffer(7)]],uint2 group [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],ushort sg [[simdgroup_index_in_threadgroup]]){
    threadgroup float left[V2_FWD_M*V2_FWD_K],right[V2_FWD_K*V2_FWD_N],result[V2_FWD_M*V2_FWD_N];
    v2_dense<V2_FWD_M,V2_FWD_N,V2_FWD_K,V2_FWD_SG,0,false>(x,w,bias,y,a,group,tid,sg,left,right,result,bias,y);
}
kernel void v2_linear_dx(device const float* dy [[buffer(0)]],device const float* w [[buffer(1)]],
                         device float* dx [[buffer(2)]],constant Args& a [[buffer(7)]],
                         uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]]){
    threadgroup float left[V2_DX_M*V2_DX_K],right[V2_DX_K*V2_DX_N],result[V2_DX_M*V2_DX_N];
    v2_dense<V2_DX_M,V2_DX_N,V2_DX_K,V2_DX_SG,1,false>(dy,w,w,dx,a,group,tid,sg,left,right,result,w,dx);
}
kernel void v2_linear_dw(device const float* x [[buffer(0)]],device const float* dy [[buffer(1)]],
                         device float* dw [[buffer(2)]],constant Args& a [[buffer(7)]],
                         uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]]){
    threadgroup float left[V2_DW_M*V2_DW_K],right[V2_DW_K*V2_DW_N],result[V2_DW_M*V2_DW_N];
    v2_dense<V2_DW_M,V2_DW_N,V2_DW_K,V2_DW_SG,2,false>(x,dy,dy,dw,a,group,tid,sg,left,right,result,dy,dw);
}

// Keep raw activations for backward/traces while applying the next elementwise
// operation before the value leaves its owning lane. Args.step: 1 ReLU, 2 add.
kernel void v2_linear_forward_epilogue(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],
                                       device const float* bias [[buffer(2)]],device float* raw [[buffer(3)]],
                                       device const float* residual [[buffer(4)]],device float* fused [[buffer(5)]],
                                       constant Args& a [[buffer(7)]],uint2 group [[threadgroup_position_in_grid]],
                                       uint tid [[thread_index_in_threadgroup]],ushort sg [[simdgroup_index_in_threadgroup]]){
    threadgroup float left[V2_FWD_M*V2_FWD_K],right[V2_FWD_K*V2_FWD_N],result[V2_FWD_M*V2_FWD_N];
    v2_dense<V2_FWD_M,V2_FWD_N,V2_FWD_K,V2_FWD_SG,0,true>(x,w,bias,raw,a,group,tid,sg,left,right,result,residual,fused);
}

// The reference arithmetic/order is available for isolated fusion candidates.
// Dispatch with the original 16x16 thread layout, not the SIMD matrix layout.
kernel void v2_reference_forward_epilogue(device const float* x [[buffer(0)]],device const float* weight [[buffer(1)]],
                                          device const float* bias [[buffer(2)]],device float* raw [[buffer(3)]],
                                          device const float* residual [[buffer(4)]],device float* fused [[buffer(5)]],
                                          constant Args& a [[buffer(7)]],uint2 group [[threadgroup_position_in_grid]],
                                          uint2 local [[thread_position_in_threadgroup]]){
    threadgroup float xx[16][16],ww[16][16];
    uint r=group.y*16+local.y,c=group.x*16+local.x;float acc=0;
    for(uint base=0;base<a.k;base+=16){
        xx[local.y][local.x]=(r<a.m&&base+local.x<a.k)?x[r*a.k+base+local.x]:0;
        uint wr=group.x*16+local.y;
        ww[local.y][local.x]=(wr<a.n&&base+local.x<a.k)?weight[wr*a.k+base+local.x]:0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<16;++j)acc=fma(xx[local.y][j],ww[local.x][j],acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(r<a.m&&c<a.n){
        float value=acc+bias[c];raw[r*a.n+c]=value;
        fused[r*a.n+c]=a.step==1?max(value,0.0f):value+residual[r*a.n+c];
    }
}

inline float2 v2_simd_ff_sum(float2 v,uint lane){
    for(uint offset=16;offset;offset>>=1){
        float2 other=float2(simd_shuffle_down(v.x,offset),simd_shuffle_down(v.y,offset));
        if(lane<offset)v=ff_add(v,other);
    }
    return float2(simd_broadcast(v.x,0),simd_broadcast(v.y,0));
}

// All lanes participate, including lanes whose element range is empty.
inline float2 v2_group_ff_sum(float2 v,threadgroup float2* scratch,uint lane,uint sg){
    v=v2_simd_ff_sum(v,lane);
    if(lane==0)scratch[sg]=v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0){
        float2 groupValue=lane<V2_NORM_SG?scratch[lane]:float2(0);
        groupValue=v2_simd_ff_sum(groupValue,lane);
        if(lane==0)scratch[0]=groupValue;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float2 total=scratch[0];
    // Prevent the next reduction from overwriting scratch before every lane reads.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

// Widths are multiples of 32 and row bases are aligned. The vector candidate
// therefore loads complete aligned float4s without crossing a logical row.
inline float4 v2_norm_load(device const float* values,uint offset){
#if V2_NORM_VEC == 4
    return reinterpret_cast<device const float4*>(values+offset)[0];
#else
    return float4(values[offset],0,0,0);
#endif
}
inline void v2_norm_store(device float* values,uint offset,float4 value){
#if V2_NORM_VEC == 4
    reinterpret_cast<device float4*>(values+offset)[0]=value;
#else
    values[offset]=value.x;
#endif
}

kernel void v2_norm_forward(device const float* x [[buffer(0)]],device const float* gamma [[buffer(1)]],
                            device const float* beta [[buffer(2)]],device float* y [[buffer(3)]],
                            device float* norm [[buffer(4)]],device float* inv [[buffer(5)]],
                            constant Args& a [[buffer(7)]],uint row [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],ushort lane [[thread_index_in_simdgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]]){
    threadgroup float2 scratch[V2_NORM_SG];
    float2 sum=0;
    for(uint base=tid*V2_NORM_VEC;base<a.w;base+=32*V2_NORM_SG*V2_NORM_VEC){
        float4 values=v2_norm_load(x,row*a.w+base);
        for(uint v=0;v<V2_NORM_VEC;++v)sum=ff_add(sum,float2(values[v],0));
    }
    float2 mean=ff_scale(v2_group_ff_sum(sum,scratch,lane,sg),1.0f/float(a.w));
    float2 variance=0;
    for(uint base=tid*V2_NORM_VEC;base<a.w;base+=32*V2_NORM_SG*V2_NORM_VEC){
        float4 values=v2_norm_load(x,row*a.w+base);
        for(uint v=0;v<V2_NORM_VEC;++v){float2 d=ff_add(float2(values[v],0),-mean);variance=ff_add(variance,ff_multiply(d,d));}
    }
    variance=ff_add(ff_scale(v2_group_ff_sum(variance,scratch,lane,sg),1.0f/float(a.w)),float2(a.epsilon,0));
    float scale=1.0f/sqrt(variance.x+variance.y);
    if(tid==0)inv[row]=scale;
    for(uint base=tid*V2_NORM_VEC;base<a.w;base+=32*V2_NORM_SG*V2_NORM_VEC){
        float4 values=v2_norm_load(x,row*a.w+base),gammas=v2_norm_load(gamma,base),betas=v2_norm_load(beta,base),zs=0,ys=0;
        for(uint v=0;v<V2_NORM_VEC;++v){
            float2 normalized=ff_scale(ff_add(float2(values[v],0),-mean),scale);
            float z=normalized.x+normalized.y;zs[v]=z;
            float2 affine=ff_add(ff_product(z,gammas[v]),float2(betas[v],0));ys[v]=affine.x+affine.y;
        }
        v2_norm_store(norm,row*a.w+base,zs);v2_norm_store(y,row*a.w+base,ys);
    }
}

kernel void v2_norm_dx(device const float* norm [[buffer(0)]],device const float* inv [[buffer(1)]],
                       device const float* gamma [[buffer(2)]],device const float* dy [[buffer(3)]],
                       device float* dx [[buffer(4)]],constant Args& a [[buffer(7)]],
                       uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]],ushort sg [[simdgroup_index_in_threadgroup]]){
    threadgroup float2 scratch[V2_NORM_SG];
    float2 sum=0,prod=0;
    for(uint base=tid*V2_NORM_VEC;base<a.w;base+=32*V2_NORM_SG*V2_NORM_VEC){
        float4 values=v2_norm_load(dy,row*a.w+base),gammas=v2_norm_load(gamma,base),norms=v2_norm_load(norm,row*a.w+base);
        for(uint v=0;v<V2_NORM_VEC;++v){float2 d=ff_product(values[v],gammas[v]);sum=ff_add(sum,d);prod=ff_add(prod,ff_scale(d,norms[v]));}
    }
    float2 mean=ff_scale(v2_group_ff_sum(sum,scratch,lane,sg),1.0f/float(a.w));
    float2 projection=ff_scale(v2_group_ff_sum(prod,scratch,lane,sg),1.0f/float(a.w));
    for(uint base=tid*V2_NORM_VEC;base<a.w;base+=32*V2_NORM_SG*V2_NORM_VEC){
        float4 values=v2_norm_load(dy,row*a.w+base),gammas=v2_norm_load(gamma,base),norms=v2_norm_load(norm,row*a.w+base),result=0;
        for(uint v=0;v<V2_NORM_VEC;++v){
            float2 d=ff_add(ff_product(values[v],gammas[v]),-mean);
            d=ff_scale(ff_add(d,-ff_scale(projection,norms[v])),inv[row]);result[v]=d.x+d.y;
        }
        v2_norm_store(dx,row*a.w+base,result);
    }
}

kernel void v2_norm_dparam(device const float* norm [[buffer(0)]],device const float* dy [[buffer(1)]],
                           device float* dg [[buffer(2)]],device float* db [[buffer(3)]],
                           constant Args& a [[buffer(7)]],uint c [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],ushort lane [[thread_index_in_simdgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]]){
    threadgroup float2 scratch[V2_NORM_SG];float2 g=0,b=0;
    for(uint r=tid;r<a.m;r+=32*V2_NORM_SG){g=ff_add(g,ff_product(norm[r*a.w+c],dy[r*a.w+c]));b=ff_add(b,float2(dy[r*a.w+c],0));}
    g=v2_group_ff_sum(g,scratch,lane,sg);b=v2_group_ff_sum(b,scratch,lane,sg);
    if(tid==0){dg[c]=g.x+g.y;db[c]=b.x+b.y;}
}

#ifndef V2_ATT_ROWS
#define V2_ATT_ROWS 1
#endif
#ifndef V2_ATT_SG
#define V2_ATT_SG 1
#endif
#ifndef V2_ATT_STAGE
#define V2_ATT_STAGE 0
#endif

// One or two SIMD groups cooperate on one query. Every lane owns a key while
// computing scores; after softmax, lanes own contiguous output channels.
// All reference trace buffers remain populated even though intermediate scores
// and probabilities are consumed from registers/threadgroup memory internally.
kernel void v2_attention_forward(device const float* qkv [[buffer(0)]],device float* scores [[buffer(1)]],
                                 device float* score_lo [[buffer(2)]],device float* p [[buffer(3)]],
                                 device float* out [[buffer(4)]],constant Args& a [[buffer(7)]],
                                 uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
                                 ushort lane [[thread_index_in_simdgroup]],ushort sg [[simdgroup_index_in_threadgroup]]){
    threadgroup float2 partial[V2_ATT_ROWS*V2_ATT_SG*32];
    threadgroup float probability[V2_ATT_ROWS*32];
#if V2_ATT_STAGE
    threadgroup float query[V2_ATT_ROWS*256];
#endif
    uint slot=sg/V2_ATT_SG,part=sg%V2_ATT_SG,rowThread=part*32+lane;
    uint q=group.x*V2_ATT_ROWS+slot,h=group.y,b=group.z,d=a.w/a.h;
    bool active=q<a.t;uint qi=(b*a.t+q)*3*a.w+h*d;
#if V2_ATT_STAGE
    for(uint c=rowThread;c<d;c+=V2_ATT_SG*32)query[slot*256+c]=active?qkv[qi+c]:0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
    float2 sums[4]={float2(0),float2(0),float2(0),float2(0)};
    if(active&&lane<=q&&lane<a.t){
        uint ki=(b*a.t+lane)*3*a.w+a.w+h*d;
        for(uint c=part;c<d;c+=V2_ATT_SG){
#if V2_ATT_STAGE
            float qv=query[slot*256+c];
#else
            float qv=qkv[qi+c];
#endif
            sums[(c/V2_ATT_SG)%4]=ff_add(sums[(c/V2_ATT_SG)%4],ff_product(qv,qkv[ki+c]));
        }
    }
    partial[(slot*V2_ATT_SG+part)*32+lane]=ff_reduce(sums);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(part==0){
        float2 score=partial[slot*V2_ATT_SG*32+lane];
        for(uint s=1;s<V2_ATT_SG;++s)score=ff_add(score,partial[(slot*V2_ATT_SG+s)*32+lane]);
        score=ff_multiply(score,float2(a.scale_hi,a.scale_lo));
        if(!active||lane>q||lane>=a.t)score=float2(-INFINITY,0);
        uint index=((b*a.h+h)*a.t+q)*a.t+lane;
        if(active&&lane<a.t){scores[index]=score.x;score_lo[index]=score.y;}
        float2 mx=score;
        for(uint offset=16;offset;offset>>=1){
            float2 other=float2(simd_shuffle_down(mx.x,offset),simd_shuffle_down(mx.y,offset));
            if(lane<offset&&(other.x>mx.x||(other.x==mx.x&&other.y>mx.y)))mx=other;
        }
        mx=float2(simd_broadcast(mx.x,0),simd_broadcast(mx.y,0));
        float e=0;
        if(active&&lane<=q&&lane<a.t){float2 centered=ff_add(score,-mx);e=exp(centered.x+centered.y);}
        float2 denominator=v2_simd_ff_sum(float2(e,0),lane);float value=0;
        if(active&&lane<=q&&lane<a.t){
            float quotient=e/denominator.x;
            float2 residual=ff_add(float2(e,0),-ff_scale(denominator,quotient));
            value=quotient+(residual.x+residual.y)/denominator.x;
        }
        probability[slot*32+lane]=value;
        if(active&&lane<a.t)p[index]=value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(active)for(uint c=rowThread;c<d;c+=V2_ATT_SG*32){
        float value=0;
        for(uint j=0;j<=q;++j)value=fma(probability[slot*32+j],qkv[(b*a.t+j)*3*a.w+2*a.w+h*d+c],value);
        out[(b*a.t+q)*a.w+h*d+c]=value;
    }
}

// dA dots reduce channels cooperatively. Each output is written by one fixed
// lane; two-stage compensated trees never depend on group scheduling.
kernel void v2_attention_da(device const float* qkv [[buffer(0)]],device const float* dout [[buffer(1)]],
                            device float* da [[buffer(2)]],device float* da_lo [[buffer(3)]],
                            constant Args& a [[buffer(7)]],uint group [[threadgroup_position_in_grid]],
                            ushort lane [[thread_index_in_simdgroup]],ushort sg [[simdgroup_index_in_threadgroup]]){
    threadgroup float2 scratch[V2_ATT_ROWS*V2_ATT_SG];
    uint slot=sg/V2_ATT_SG,part=sg%V2_ATT_SG,index=group*V2_ATT_ROWS+slot;
    bool active=index<a.b*a.h*a.t*a.t;uint j=index%a.t,q=(index/a.t)%a.t,h=(index/(a.t*a.t))%a.h,b=index/(a.h*a.t*a.t),d=a.w/a.h;
    float2 sum=0;
    if(active&&j<=q)for(uint c=part*32+lane;c<d;c+=V2_ATT_SG*32)
        sum=ff_add(sum,ff_product(dout[(b*a.t+q)*a.w+h*d+c],qkv[(b*a.t+j)*3*a.w+2*a.w+h*d+c]));
    sum=v2_simd_ff_sum(sum,lane);
    if(lane==0)scratch[slot*V2_ATT_SG+part]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(part==0&&lane==0&&active){
        sum=scratch[slot*V2_ATT_SG];for(uint s=1;s<V2_ATT_SG;++s)sum=ff_add(sum,scratch[slot*V2_ATT_SG+s]);
        da[index]=sum.x;da_lo[index]=sum.y;
    }
}

kernel void v2_attention_ds(device const float* da [[buffer(0)]],device const float* da_lo [[buffer(1)]],
                            device const float* p [[buffer(2)]],device float* ds [[buffer(3)]],
                            device float* ds_lo [[buffer(4)]],constant Args& a [[buffer(7)]],
                            uint group [[threadgroup_position_in_grid]],ushort lane [[thread_index_in_simdgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]]){
    uint slot=sg/V2_ATT_SG,part=sg%V2_ATT_SG,row=group*V2_ATT_ROWS+slot;
    // T <= 32 needs exactly one SIMD group for the softmax contraction.
    if(part!=0||row>=a.b*a.h*a.t)return;
    uint q=row%a.t,index=row*a.t+lane;float2 value=0;
    if(lane<=q&&lane<a.t)value=ff_scale(float2(da[index],da_lo[index]),p[index]);
    float2 sum=v2_simd_ff_sum(value,lane);value=0;
    if(lane<=q&&lane<a.t)value=ff_multiply(ff_scale(ff_add(float2(da[index],da_lo[index]),-sum),p[index]),float2(a.scale_hi,a.scale_lo));
    if(lane<a.t){ds[index]=value.x;ds_lo[index]=value.y;}
}

// Fuse dA = dOutput V^T with the softmax-vector product. dA remains visible in
// diagnostic buffers, but dS consumes the register value rather than rereading
// dA from device memory. This kernel does not itself recompute forward P.
kernel void v2_attention_dads(device const float* qkv [[buffer(0)]],device const float* dout [[buffer(1)]],
                              device const float* p [[buffer(2)]],device float* da [[buffer(3)]],
                              device float* da_lo [[buffer(4)]],device float* ds [[buffer(5)]],
                              device float* ds_lo [[buffer(6)]],constant Args& a [[buffer(7)]],
                              uint group [[threadgroup_position_in_grid]],ushort lane [[thread_index_in_simdgroup]],
                              ushort sg [[simdgroup_index_in_threadgroup]]){
    threadgroup float2 partial[V2_ATT_ROWS*V2_ATT_SG*32];
    uint slot=sg/V2_ATT_SG,part=sg%V2_ATT_SG,row=group*V2_ATT_ROWS+slot;
    bool active=row<a.b*a.h*a.t;uint q=row%a.t,h=(row/a.t)%a.h,b=row/(a.h*a.t),d=a.w/a.h;
    float2 sums[4]={float2(0),float2(0),float2(0),float2(0)};
    if(active&&lane<=q&&lane<a.t)for(uint c=part;c<d;c+=V2_ATT_SG){
        float productA=dout[(b*a.t+q)*a.w+h*d+c],productB=qkv[(b*a.t+lane)*3*a.w+2*a.w+h*d+c];
        sums[(c/V2_ATT_SG)%4]=ff_add(sums[(c/V2_ATT_SG)%4],ff_product(productA,productB));
    }
    partial[(slot*V2_ATT_SG+part)*32+lane]=ff_reduce(sums);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(part==0){
        float2 value=partial[slot*V2_ATT_SG*32+lane];
        for(uint s=1;s<V2_ATT_SG;++s)value=ff_add(value,partial[(slot*V2_ATT_SG+s)*32+lane]);
        uint index=row*a.t+lane;
        if(active&&lane<a.t){da[index]=value.x;da_lo[index]=value.y;}
        float2 weighted=0;
        if(active&&lane<=q&&lane<a.t)weighted=ff_scale(value,p[index]);
        float2 sum=v2_simd_ff_sum(weighted,lane),gradient=0;
        if(active&&lane<=q&&lane<a.t)gradient=ff_multiply(ff_scale(ff_add(value,-sum),p[index]),float2(a.scale_hi,a.scale_lo));
        if(active&&lane<a.t){ds[index]=gradient.x;ds_lo[index]=gradient.y;}
    }
}

// Build an ascending row-index list once per token, sharing it across all W
// gradient channels. Integer prefix sums fix the list order independently of
// GPU group scheduling. indices has 128*M elements; counts has 128 elements.
#ifndef V2_EMBED_BLOCK
#define V2_EMBED_BLOCK 64
#endif
kernel void v2_embedding_index(device const int* tokens [[buffer(0)]],device uint* indices [[buffer(1)]],
                                device uint* counts [[buffer(2)]],constant Args& a [[buffer(7)]],
                                uint token [[threadgroup_position_in_grid]],ushort lane [[thread_index_in_simdgroup]]){
    constexpr uint perLane=V2_EMBED_BLOCK/32;uint total=0;
    for(uint base=0;base<a.m;base+=V2_EMBED_BLOCK){
        uint selected[perLane],count=0;
        for(uint j=0;j<perLane;++j){uint row=base+lane*perLane+j;
            selected[j]=(row<a.m&&uint(tokens[row])==token)?1u:0u;count+=selected[j];}
        uint prefix=count;
        for(uint offset=1;offset<32;offset<<=1){uint earlier=simd_shuffle_up(prefix,offset);if(lane>=offset)prefix+=earlier;}
        uint offset=total+prefix-count;
        for(uint j=0;j<perLane;++j)if(selected[j])indices[token*a.m+offset++]=base+lane*perLane+j;
        total+=simd_broadcast(prefix,31);
    }
    if(lane==0)counts[token]=total;
}

kernel void v2_embedding_grad(device const uint* indices [[buffer(0)]],device const uint* counts [[buffer(1)]],
                               device const float* dx [[buffer(2)]],device float* dt [[buffer(3)]],
                               device float* dp [[buffer(4)]],constant Args& a [[buffer(7)]],
                               uint i [[thread_position_in_grid]]){
    if(i<128*a.w){
        uint token=i/a.w,c=i%a.w;float sum=0;
        for(uint j=0;j<counts[token];++j)sum+=dx[indices[token*a.m+j]*a.w+c];
        dt[i]=sum;
    }
    if(i<32*a.w){
        uint pos=i/a.w,c=i%a.w;float sum=0;
        if(pos<a.t)for(uint b=0;b<a.b;++b)sum+=dx[(b*a.t+pos)*a.w+c];
        dp[i]=sum;
    }
}

// Tensor update expressions match the reference Adam kernel. The only atomic
// operation combines integer failure flags; no floating reduction is atomic.
kernel void v2_adam_finite(device float* p [[buffer(0)]],device const float* g [[buffer(1)]],
                           device float* m [[buffer(2)]],device float* v [[buffer(3)]],
                           device atomic_uint* flag [[buffer(4)]],constant Args& a [[buffer(7)]],
                           uint i [[thread_position_in_grid]]){
    if(i>=a.m)return;
    float mm=m[i]+0.1f*(g[i]-m[i]);
    float vv=a.beta2*v[i]+0.001f*(g[i]*g[i]);
    m[i]=mm;v[i]=vv;p[i]-=a.lr*(mm/a.corr1)/(sqrt(vv/a.corr2)+a.epsilon);
    if(!isfinite(p[i])||!isfinite(mm)||!isfinite(vv)||!isfinite(g[i]))
        atomic_fetch_or_explicit(flag,1u,memory_order_relaxed);
}
