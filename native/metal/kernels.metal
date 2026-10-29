#include <metal_stdlib>
using namespace metal;

// The host passes dimensions, never model math. All arrays store FP32 values.
struct Args {
    uint m, n, k, b, t, w, h, step;
    float epsilon, lr, beta1, beta2, corr1, corr2, scale_hi, scale_lo;
};

// Software two-float expansion using FP32 instructions, not native FP64.
// Safe compiler math preserves sum decomposition and FMA product residuals.
inline float2 ff_add(float2 a,float2 b){
    float s=a.x+b.x,v=s-a.x;
    float error=(a.x-(s-v))+(b.x-v);
    float tail=(a.y+b.y)+error;
    float hi=s+tail,z=hi-s;
    return float2(hi,(s-(hi-z))+(tail-z));
}
inline float2 ff_product(float a,float b){float p=a*b;return float2(p,fma(a,b,-p));}
inline float2 ff_scale(float2 a,float b){return ff_add(ff_product(a.x,b),ff_product(a.y,b));}
inline float2 ff_multiply(float2 a,float2 b){return ff_add(ff_scale(a,b.x),ff_scale(a,b.y));}
inline float2 ff_reduce(thread float2* s){return ff_add(ff_add(s[0],s[1]),ff_add(s[2],s[3]));}

kernel void gather_rows(device const int *rows [[buffer(0)]],
                        device const int *indices [[buffer(1)]],
                        device int *tokens [[buffer(2)]], device int *targets [[buffer(3)]],
                        constant Args &a [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.b*a.t) return;
    uint r = uint(indices[a.k+i/a.t]);
    tokens[i] = rows[r*(a.t+1)+i%a.t];
    targets[i] = rows[r*(a.t+1)+i%a.t+1];
}

kernel void embedding(device const int *tokens [[buffer(0)]],
                      device const float *table [[buffer(1)]], device const float *position [[buffer(2)]],
                      device float *x [[buffer(3)]], constant Args &a [[buffer(7)]],
                      uint i [[thread_position_in_grid]]) {
    if (i < a.m*a.w) x[i] = table[uint(tokens[i/a.w])*a.w+i%a.w] + position[(i/a.w%a.t)*a.w+i%a.w];
}

// A distinct writer owns each shared embedding gradient. Repeated tokens are
// reduced in ascending row order, with no atomics or schedule-dependent sums.
kernel void embedding_grad(device const int *tokens [[buffer(0)]], device const float *dx [[buffer(1)]],
                           device float *dt [[buffer(2)]], device float *dp [[buffer(3)]],
                           constant Args &a [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i < 128*a.w) {
        uint tok=i/a.w, c=i%a.w; float s=0;
        for (uint r=0;r<a.m;++r) if (uint(tokens[r])==tok) s += dx[r*a.w+c];
        dt[i]=s;
    }
    if (i < 32*a.w) {
        uint pos=i/a.w,c=i%a.w; float s=0;
        if (pos<a.t) for (uint b=0;b<a.b;++b) s += dx[(b*a.t+pos)*a.w+c];
        dp[i]=s;
    }
}

// Matrix C[M,N]=A[M,K]B[N,K]^T. One 16x16 output tile per group.
// The K loop is ascending and every output has exactly one writer.
kernel void linear_forward(device const float *x [[buffer(0)]], device const float *weight [[buffer(1)]],
                           device const float *bias [[buffer(2)]], device float *y [[buffer(3)]],
                           constant Args &a [[buffer(7)]], uint2 g [[threadgroup_position_in_grid]],
                           uint2 l [[thread_position_in_threadgroup]]) {
    threadgroup float xx[16][16], ww[16][16];
    uint r=g.y*16+l.y, c=g.x*16+l.x;
    float acc=0;
    for (uint base=0;base<a.k;base+=16) {
        xx[l.y][l.x]=(r<a.m && base+l.x<a.k)?x[r*a.k+base+l.x]:0;
        uint wr=g.x*16+l.y;
        ww[l.y][l.x]=(wr<a.n && base+l.x<a.k)?weight[wr*a.k+base+l.x]:0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j=0;j<16;++j) acc=fma(xx[l.y][j],ww[l.x][j],acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (r<a.m && c<a.n) y[r*a.n+c]=acc+bias[c];
}

kernel void linear_dx(device const float *dy [[buffer(0)]], device const float *weight [[buffer(1)]],
                      device float *dx [[buffer(2)]], constant Args &a [[buffer(7)]],
                      uint2 g [[threadgroup_position_in_grid]],uint2 l [[thread_position_in_threadgroup]]) {
    threadgroup float dd[16][16], ww[16][16];
    uint r=g.y*16+l.y,c=g.x*16+l.x;float acc=0;
    for(uint base=0;base<a.n;base+=16){
        dd[l.y][l.x]=(r<a.m && base+l.x<a.n)?dy[r*a.n+base+l.x]:0;
        ww[l.y][l.x]=(base+l.y<a.n && c<a.k)?weight[(base+l.y)*a.k+c]:0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<16;++j)acc=fma(dd[l.y][j],ww[j][l.x],acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(r<a.m && c<a.k)dx[r*a.k+c]=acc;
}

// Selective compensated QKV contractions, motivated by propagated gradient
// failures after the first normalization. Other dense kernels are unchanged.
kernel void qkv_forward(device const float *x [[buffer(0)]],device const float *weight [[buffer(1)]],
                        device const float *bias [[buffer(2)]],device float *y [[buffer(3)]],
                        constant Args &a [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],
                        uint2 l [[thread_position_in_threadgroup]]){
    threadgroup float xx[16][16],ww[16][16];
    uint r=g.y*16+l.y,c=g.x*16+l.x;float2 acc=0;
    for(uint base=0;base<a.k;base+=16){
        xx[l.y][l.x]=(r<a.m&&base+l.x<a.k)?x[r*a.k+base+l.x]:0;
        uint wr=g.x*16+l.y;ww[l.y][l.x]=(wr<a.n&&base+l.x<a.k)?weight[wr*a.k+base+l.x]:0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<16;++j)acc=ff_add(acc,ff_product(xx[l.y][j],ww[l.x][j]));
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(r<a.m&&c<a.n){acc=ff_add(acc,float2(bias[c],0));y[r*a.n+c]=acc.x+acc.y;}
}

kernel void qkv_dx(device const float *dy [[buffer(0)]],device const float *weight [[buffer(1)]],
                   device float *dx [[buffer(2)]],constant Args &a [[buffer(7)]],
                   uint2 g [[threadgroup_position_in_grid]],uint2 l [[thread_position_in_threadgroup]]){
    threadgroup float dd[16][16],ww[16][16];
    uint r=g.y*16+l.y,c=g.x*16+l.x;float2 acc=0;
    for(uint base=0;base<a.n;base+=16){
        dd[l.y][l.x]=(r<a.m&&base+l.x<a.n)?dy[r*a.n+base+l.x]:0;
        ww[l.y][l.x]=(base+l.y<a.n&&c<a.k)?weight[(base+l.y)*a.k+c]:0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<16;++j)acc=ff_add(acc,ff_product(dd[l.y][j],ww[j][l.x]));
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(r<a.m&&c<a.k)dx[r*a.k+c]=acc.x+acc.y;
}

kernel void linear_dw(device const float *x [[buffer(0)]], device const float *dy [[buffer(1)]],
                      device float *dw [[buffer(2)]], constant Args &a [[buffer(7)]],
                      uint2 g [[threadgroup_position_in_grid]],uint2 l [[thread_position_in_threadgroup]]) {
    threadgroup float xx[16][16], dd[16][16];
    uint r=g.y*16+l.y,c=g.x*16+l.x;float acc=0;
    for(uint base=0;base<a.m;base+=16){
        xx[l.y][l.x]=(base+l.y<a.m && c<a.k)?x[(base+l.y)*a.k+c]:0;
        uint dr=g.y*16+l.x;
        dd[l.y][l.x]=(base+l.y<a.m && dr<a.n)?dy[(base+l.y)*a.n+dr]:0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<16;++j)acc=fma(dd[j][l.y],xx[j][l.x],acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(r<a.n && c<a.k)dw[r*a.k+c]=acc;
}

kernel void linear_db(device const float *dy [[buffer(0)]],device float *db [[buffer(1)]],
                      constant Args &a [[buffer(7)]],uint c [[thread_position_in_grid]]){
    if(c>=a.n)return;float4 s=0;
    for(uint r=0;r<a.m;++r)s[r%4]+=dy[r*a.n+c];
    db[c]=(s.x+s.y)+(s.z+s.w);
}

kernel void norm_forward(device const float *x [[buffer(0)]],device const float *gamma [[buffer(1)]],
                         device const float *beta [[buffer(2)]],device float *y [[buffer(3)]],
                         device float *norm [[buffer(4)]],device float *inv [[buffer(5)]],
                         constant Args &a [[buffer(7)]],uint r [[thread_position_in_grid]]){
    if(r>=a.m)return;
    float2 sum[4]={float2(0),float2(0),float2(0),float2(0)},var[4]={float2(0),float2(0),float2(0),float2(0)};
    for(uint c=0;c<a.w;++c)sum[c%4]=ff_add(sum[c%4],float2(x[r*a.w+c],0));
    float2 mean=ff_scale(ff_reduce(sum),1.0f/float(a.w));
    for(uint c=0;c<a.w;++c){float2 d=ff_add(float2(x[r*a.w+c],0),-mean);var[c%4]=ff_add(var[c%4],ff_multiply(d,d));}
    float2 variance=ff_add(ff_scale(ff_reduce(var),1.0f/float(a.w)),float2(a.epsilon,0));
    float scale=1.0f/sqrt(variance.x+variance.y);
    inv[r]=scale;
    for(uint c=0;c<a.w;++c){float2 normalized=ff_scale(ff_add(float2(x[r*a.w+c],0),-mean),scale);
        float z=normalized.x+normalized.y;norm[r*a.w+c]=z;
        float2 affine=ff_add(ff_product(z,gamma[c]),float2(beta[c],0));y[r*a.w+c]=affine.x+affine.y;}
}

kernel void norm_dx(device const float *norm [[buffer(0)]],device const float *inv [[buffer(1)]],
                    device const float *gamma [[buffer(2)]],device const float *dy [[buffer(3)]],
                    device float *dx [[buffer(4)]],constant Args &a [[buffer(7)]],uint r [[thread_position_in_grid]]){
    if(r>=a.m)return;float2 sum[4]={float2(0),float2(0),float2(0),float2(0)},prod[4]={float2(0),float2(0),float2(0),float2(0)};
    for(uint c=0;c<a.w;++c){float2 d=ff_product(dy[r*a.w+c],gamma[c]);sum[c%4]=ff_add(sum[c%4],d);prod[c%4]=ff_add(prod[c%4],ff_scale(d,norm[r*a.w+c]));}
    float2 mean=ff_scale(ff_reduce(sum),1.0f/float(a.w)),projection=ff_scale(ff_reduce(prod),1.0f/float(a.w));
    for(uint c=0;c<a.w;++c){float2 d=ff_add(ff_product(dy[r*a.w+c],gamma[c]),-mean);
        d=ff_scale(ff_add(d,-ff_scale(projection,norm[r*a.w+c])),inv[r]);dx[r*a.w+c]=d.x+d.y;}
}

kernel void norm_dparam(device const float *norm [[buffer(0)]],device const float *dy [[buffer(1)]],
                        device float *dg [[buffer(2)]],device float *db [[buffer(3)]],
                        constant Args &a [[buffer(7)]],uint c [[thread_position_in_grid]]){
    if(c>=a.w)return;float2 g[4]={float2(0),float2(0),float2(0),float2(0)},b[4]={float2(0),float2(0),float2(0),float2(0)};
    for(uint r=0;r<a.m;++r){g[r%4]=ff_add(g[r%4],ff_product(norm[r*a.w+c],dy[r*a.w+c]));b[r%4]=ff_add(b[r%4],float2(dy[r*a.w+c],0));}
    float2 gg=ff_reduce(g),bb=ff_reduce(b);dg[c]=gg.x+gg.y;db[c]=bb.x+bb.y;
}

kernel void add(device const float *x [[buffer(0)]],device const float *y [[buffer(1)]],device float *z [[buffer(2)]],
                constant Args &a [[buffer(7)]],uint i [[thread_position_in_grid]]){if(i<a.m)z[i]=x[i]+y[i];}
kernel void relu_forward(device const float *x [[buffer(0)]],device float *y [[buffer(1)]],
                         constant Args &a [[buffer(7)]],uint i [[thread_position_in_grid]]){if(i<a.m)y[i]=max(x[i],0.0f);}
kernel void relu_backward(device const float *x [[buffer(0)]],device const float *dy [[buffer(1)]],device float *dx [[buffer(2)]],
                          constant Args &a [[buffer(7)]],uint i [[thread_position_in_grid]]){if(i<a.m)dx[i]=x[i]>0?dy[i]:0;}

kernel void attention_scores(device const float *qkv [[buffer(0)]],device float *scores [[buffer(1)]],device float *score_lo [[buffer(2)]],
                             constant Args &a [[buffer(7)]],uint i [[thread_position_in_grid]]){
    if(i>=a.b*a.h*a.t*a.t)return;
    uint j=i%a.t,q=(i/a.t)%a.t,h=(i/(a.t*a.t))%a.h,b=i/(a.h*a.t*a.t),d=a.w/a.h;
    if(j>q){scores[i]=-INFINITY;score_lo[i]=0;return;}
    uint qi=(b*a.t+q)*3*a.w+h*d,ki=(b*a.t+j)*3*a.w+a.w+h*d;
    float2 s[4]={float2(0),float2(0),float2(0),float2(0)};
    for(uint c=0;c<d;++c)s[c%4]=ff_add(s[c%4],ff_product(qkv[qi+c],qkv[ki+c]));
    float2 score=ff_multiply(ff_reduce(s),float2(a.scale_hi,a.scale_lo));
    scores[i]=score.x;score_lo[i]=score.y;
}

kernel void attention_softmax(device const float *scores [[buffer(0)]],device const float *score_lo [[buffer(1)]],device float *p [[buffer(2)]],
                              constant Args &a [[buffer(7)]],uint r [[thread_position_in_grid]]){
    if(r>=a.b*a.h*a.t)return;uint q=r%a.t;float2 mx=float2(-INFINITY,0);
    for(uint j=0;j<=q;++j){float2 v=float2(scores[r*a.t+j],score_lo[r*a.t+j]);if(v.x>mx.x||(v.x==mx.x&&v.y>mx.y))mx=v;}
    float2 sum=0;for(uint j=0;j<=q;++j){float2 centered=ff_add(float2(scores[r*a.t+j],score_lo[r*a.t+j]),-mx);
        float e=exp(centered.x+centered.y);p[r*a.t+j]=e;sum=ff_add(sum,float2(e,0));}
    for(uint j=0;j<a.t;++j){float probability=0;
        if(j<=q){float e=p[r*a.t+j],quotient=e/sum.x;
            float2 residual=ff_add(float2(e,0),-ff_scale(sum,quotient));
            probability=quotient+(residual.x+residual.y)/sum.x;}
        p[r*a.t+j]=probability;}
}

kernel void attention_values(device const float *qkv [[buffer(0)]],device const float *p [[buffer(1)]],
                             device float *out [[buffer(2)]],constant Args &a [[buffer(7)]],uint i [[thread_position_in_grid]]){
    if(i>=a.b*a.t*a.w)return;
    uint c=i%a.w,q=(i/a.w)%a.t,b=i/(a.w*a.t),h=c/(a.w/a.h);float s=0;
    for(uint j=0;j<=q;++j)s=fma(p[((b*a.h+h)*a.t+q)*a.t+j],qkv[(b*a.t+j)*3*a.w+2*a.w+c],s);
    out[i]=s;
}

kernel void attention_da(device const float *qkv [[buffer(0)]],device const float *dout [[buffer(1)]],
                         device float *da [[buffer(2)]],device float *da_lo [[buffer(3)]],constant Args &a [[buffer(7)]],uint i [[thread_position_in_grid]]){
    if(i>=a.b*a.h*a.t*a.t)return;
    uint j=i%a.t,q=(i/a.t)%a.t,h=(i/(a.t*a.t))%a.h,b=i/(a.h*a.t*a.t),d=a.w/a.h;
    if(j>q){da[i]=0;da_lo[i]=0;return;}float2 s[4]={float2(0),float2(0),float2(0),float2(0)};
    for(uint c=0;c<d;++c)s[c%4]=ff_add(s[c%4],ff_product(dout[(b*a.t+q)*a.w+h*d+c],qkv[(b*a.t+j)*3*a.w+2*a.w+h*d+c]));
    float2 value=ff_reduce(s);da[i]=value.x;da_lo[i]=value.y;
}

kernel void attention_ds(device const float *da [[buffer(0)]],device const float *da_lo [[buffer(1)]],device const float *p [[buffer(2)]],
                         device float *ds [[buffer(3)]],device float *ds_lo [[buffer(4)]],constant Args &a [[buffer(7)]],uint r [[thread_position_in_grid]]){
    if(r>=a.b*a.h*a.t)return;uint q=r%a.t;float2 s[4]={float2(0),float2(0),float2(0),float2(0)};
    for(uint j=0;j<=q;++j)s[j%4]=ff_add(s[j%4],ff_scale(float2(da[r*a.t+j],da_lo[r*a.t+j]),p[r*a.t+j]));
    float2 sum=ff_reduce(s),scale=float2(a.scale_hi,a.scale_lo);
    for(uint j=0;j<a.t;++j){float2 value=0;
        if(j<=q)value=ff_multiply(ff_scale(ff_add(float2(da[r*a.t+j],da_lo[r*a.t+j]),-sum),p[r*a.t+j]),scale);
        ds[r*a.t+j]=value.x;ds_lo[r*a.t+j]=value.y;}
}

kernel void attention_dqkv(device const float *qkv [[buffer(0)]],device const float *p [[buffer(1)]],
                           device const float *dout [[buffer(2)]],device const float *ds [[buffer(3)]],
                           device const float *ds_lo [[buffer(4)]],device float *dqkv [[buffer(5)]],constant Args &a [[buffer(7)]],uint i [[thread_position_in_grid]]){
    if(i>=a.b*a.t*3*a.w)return;
    uint c=i%a.w,which=(i/a.w)%3,pos=(i/(3*a.w))%a.t,b=i/(a.t*3*a.w),h=c/(a.w/a.h);
    float2 s[4]={float2(0),float2(0),float2(0),float2(0)};
    if(which==0){for(uint j=0;j<=pos;++j){uint z=((b*a.h+h)*a.t+pos)*a.t+j;s[j%4]=ff_add(s[j%4],ff_scale(float2(ds[z],ds_lo[z]),qkv[(b*a.t+j)*3*a.w+a.w+c]));}}
    else if(which==1){for(uint q=pos;q<a.t;++q){uint z=((b*a.h+h)*a.t+q)*a.t+pos;s[q%4]=ff_add(s[q%4],ff_scale(float2(ds[z],ds_lo[z]),qkv[(b*a.t+q)*3*a.w+c]));}}
    else{for(uint q=pos;q<a.t;++q)s[q%4]=ff_add(s[q%4],ff_product(p[((b*a.h+h)*a.t+q)*a.t+pos],dout[(b*a.t+q)*a.w+c]));}
    float2 value=ff_reduce(s);dqkv[i]=value.x+value.y;
}

kernel void cross_entropy(device const float *logits [[buffer(0)]],device const int *targets [[buffer(1)]],
                          device float *dlogits [[buffer(2)]],device float *rowloss [[buffer(3)]],
                          constant Args &a [[buffer(7)]],uint r [[thread_position_in_grid]]){
    if(r>=a.m)return;float mx=-INFINITY;for(uint c=0;c<128;++c)mx=max(mx,logits[r*128+c]);
    float4 s=0;for(uint c=0;c<128;++c){float e=exp(logits[r*128+c]-mx);dlogits[r*128+c]=e;s[c%4]+=e;}
    float sum=(s.x+s.y)+(s.z+s.w);rowloss[r]=log(sum)+(mx-logits[r*128+uint(targets[r])]);
    for(uint c=0;c<128;++c)dlogits[r*128+c]=(dlogits[r*128+c]/sum-(c==uint(targets[r])?1.0f:0.0f))/float(a.m);
}
kernel void loss_mean(device const float *rowloss [[buffer(0)]],device float *loss [[buffer(1)]],
                      constant Args &a [[buffer(7)]],uint i [[thread_position_in_grid]]){
    if(i)return;float4 s=0;for(uint r=0;r<a.m;++r)s[r%4]+=rowloss[r];loss[0]=((s.x+s.y)+(s.z+s.w))/float(a.m);
}
kernel void adam(device float *p [[buffer(0)]],device const float *g [[buffer(1)]],device float *m [[buffer(2)]],device float *v [[buffer(3)]],
                 constant Args &a [[buffer(7)]],uint i [[thread_position_in_grid]]){
    if(i>=a.m)return;
    // Fixed mathematical complements are rounded once, as in the CPU contract.
    float mm=m[i]+0.1f*(g[i]-m[i]);
    float vv=a.beta2*v[i]+0.001f*(g[i]*g[i]);
    m[i]=mm;v[i]=vv;p[i]-=a.lr*(mm/a.corr1)/(sqrt(vv/a.corr2)+a.epsilon);
}
kernel void finite_reset(device atomic_uint *flag [[buffer(0)]],uint i [[thread_position_in_grid]]){
    if(i==0)atomic_store_explicit(flag,0u,memory_order_relaxed);
}
kernel void finite_scan(device const float *values [[buffer(0)]],device atomic_uint *flag [[buffer(1)]],
                        constant Args &a [[buffer(7)]],uint i [[thread_position_in_grid]]){
    if(i<a.m && !isfinite(values[i]))atomic_fetch_or_explicit(flag,1u,memory_order_relaxed);
}
