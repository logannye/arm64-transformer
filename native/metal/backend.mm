#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef TFM_HOST_SHA256
#error "Use the provided Makefile to embed host source provenance"
#endif

namespace {
thread_local std::string lastError;
constexpr size_t GUARD=256;
constexpr uint32_t MARK=0x5a6bc7d8;
struct Args {uint32_t m=0,n=0,k=0,b=0,t=0,w=0,h=4,step=0;
    float epsilon=1e-5f,lr=.001f,beta1=.9f,beta2=.999f,corr1=1,corr2=1,scale_hi=0,scale_lo=0;};
struct Buffer {
    id<MTLBuffer> object; size_t bytes=0;
    Buffer(id<MTLDevice> device,size_t n):bytes(n) {
        object=[device newBufferWithLength:n+2*GUARD options:MTLResourceStorageModeShared];
        if(!object)throw std::runtime_error("Metal buffer allocation failed");
        std::memset(object.contents,0,n+2*GUARD);
        for(size_t i=0;i<GUARD/4;++i){static_cast<uint32_t*>(object.contents)[i]=MARK;
            reinterpret_cast<uint32_t*>(static_cast<char*>(object.contents)+GUARD+n)[i]=MARK;}
    }
    void* data(){return static_cast<char*>(object.contents)+GUARD;}
    bool guards(){for(size_t i=0;i<GUARD/4;++i)if(static_cast<uint32_t*>(object.contents)[i]!=MARK ||
        reinterpret_cast<uint32_t*>(static_cast<char*>(object.contents)+GUARD+bytes)[i]!=MARK)return false;return true;}
};
using Buf=std::shared_ptr<Buffer>;
struct Ref {Buf buffer;size_t offset=0;};
struct Input {Buf values;uint32_t rows=0,t=0,kind=0;std::vector<int32_t> indices;};
struct Batch {Buf tokens,targets;uint32_t b,t;bool hasTargets;};
enum P {TOKEN,POSITION,LN1W,LN1B,QKVW,QKVB,ATTW,ATTB,LN2W,LN2B,FF1W,FF1B,FF2W,FF2B,LN3W,LN3B,HEADW,HEADB,NP};

struct Context {
    id<MTLDevice> device;id<MTLCommandQueue> queue;id<MTLLibrary> library;
    std::map<std::string,id<MTLComputePipelineState>> pipelines;
    std::map<std::string,Buf> buffers;
    std::map<uint64_t,Input> inputs;std::map<uint64_t,Batch> batches;
    std::vector<id<MTLCommandBuffer>> pending;
    uint32_t w,maxB,maxT,step=0,lastB=0,lastT=0,lastMode=0;
    float attentionScaleHi=0,attentionScaleLo=0;
    uint64_t nextHandle=1;size_t offsets[NP],sizes[NP],paramCount=0,dispatchCount=0,commandCount=0;
    double lastGPUTime=0;std::string receipt;
    explicit Context(uint32_t width,uint32_t mb,uint32_t mt,const char* source,const float* initial,size_t count):w(width),maxB(mb),maxT(mt) {
        if(!(w==32||w==64||w==128||w==256||w==512||w==1024)||mb<1||mb>16||mt<1||mt>32)
            throw std::runtime_error("Unsupported width/batch/context limits");
        double attentionScale=1.0/std::sqrt(double(w/4));
        attentionScaleHi=float(attentionScale);attentionScaleLo=float(attentionScale-double(attentionScaleHi));
        size_t lengths[NP]={128*w,32*w,w,w,3*w*w,3*w,w*w,w,w,w,2*w*w,2*w,2*w*w,w,w,w,128*w,128};
        for(int i=0;i<NP;++i){offsets[i]=paramCount;sizes[i]=lengths[i];paramCount+=lengths[i];}
        if(count!=paramCount||!initial)throw std::runtime_error("Initial parameter count mismatch");
        for(size_t i=0;i<count;++i)if(!std::isfinite(initial[i]))throw std::runtime_error("Nonfinite initial parameters");
        device=MTLCreateSystemDefaultDevice();if(!device)throw std::runtime_error("No Metal device");
        if(!device.hasUnifiedMemory)throw std::runtime_error("This backend requires Apple unified memory");
        queue=[device newCommandQueue];if(!queue)throw std::runtime_error("Cannot create Metal command queue");
        NSError* error=nil;MTLCompileOptions* options=[MTLCompileOptions new];
        options.mathMode=MTLMathModeSafe;options.mathFloatingPointFunctions=MTLMathFloatingPointFunctionsPrecise;
        options.languageVersion=MTLLanguageVersion3_1;
        NSString* code=[NSString stringWithUTF8String:source];
        if(!code)throw std::runtime_error("Invalid shader UTF-8");
        library=[device newLibraryWithSource:code options:options error:&error];
        if(!library)throw std::runtime_error(std::string("Metal runtime shader compilation: ")+error.localizedDescription.UTF8String);
        const char* names[]={"gather_rows","embedding","embedding_grad","linear_forward","linear_dx","linear_dw","linear_db","qkv_forward","qkv_dx",
            "norm_forward","norm_dx","norm_dparam","add","relu_forward","relu_backward","attention_scores","attention_softmax",
            "attention_values","attention_da","attention_ds","attention_dqkv","cross_entropy","loss_mean","adam","finite_reset","finite_scan"};
        for(auto name:names){id<MTLFunction> f=[library newFunctionWithName:[NSString stringWithUTF8String:name]];
            id<MTLComputePipelineState> p=[device newComputePipelineStateWithFunction:f error:&error];
            if(!p)throw std::runtime_error(std::string("Pipeline ")+name+": "+error.localizedDescription.UTF8String);
            if(p.maxTotalThreadsPerThreadgroup<256)throw std::runtime_error("Pipeline does not support 256-thread tile");
            pipelines[name]=p;}
        for(auto name:{"parameters","gradients","m","v"})allocate(name,paramCount);
        std::memcpy(buffers.at("parameters")->data(),initial,count*4);
        size_t rows=mb*mt;
        for(auto name:{"x","n1","attention","projected","r1","n2","ff2","r2","n3","norm1","norm2","norm3",
                       "dn3","dr2","dr1","dn2","temp","dattention","dn1","dx"})allocate(name,rows*w);
        for(auto name:{"ff1","relu","drelu","dff1"})allocate(name,rows*2*w);
        for(auto name:{"qkv","dqkv"})allocate(name,rows*3*w);
        for(auto name:{"scores","score_lo","probs","da","da_lo","ds","ds_lo"})allocate(name,mb*4*mt*mt);
        for(auto name:{"logits","dlogits"})allocate(name,rows*128);
        for(auto name:{"inv1","inv2","inv3","rowloss"})allocate(name,rows);
        allocate("loss",1);allocate("finite",1);
    }
    ~Context(){for(auto c:pending)[c waitUntilCompleted];}
    void allocate(const std::string& n,size_t floats){buffers[n]=std::make_shared<Buffer>(device,floats*4);}
    Ref r(const std::string& n){return {buffers.at(n),0};}
    Ref p(P index,bool grad=false){return {buffers.at(grad?"gradients":"parameters"),offsets[index]*4};}
    Args args(uint32_t b,uint32_t t){Args a;a.b=b;a.t=t;a.m=b*t;a.w=w;
        a.scale_hi=attentionScaleHi;a.scale_lo=attentionScaleLo;return a;}
    void sync(){lastGPUTime=0;for(auto c:pending){[c waitUntilCompleted];
        if(c.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(std::string("GPU command failed: ")+c.error.localizedDescription.UTF8String);
        if(c.GPUEndTime>=c.GPUStartTime)lastGPUTime+=c.GPUEndTime-c.GPUStartTime;}pending.clear();}
    id<MTLCommandBuffer> command(){id<MTLCommandBuffer> c=[queue commandBuffer];if(!c)throw std::runtime_error("Cannot allocate command buffer");++commandCount;return c;}
    void commit(id<MTLCommandBuffer> c){[c commit];pending.push_back(c);}
    void encode(id<MTLComputeCommandEncoder> e,const char* name,std::initializer_list<Ref> refs,const Args& a,size_t count){
        auto pipe=pipelines.at(name);[e setComputePipelineState:pipe];uint32_t slot=0;
        for(auto ref:refs){[e setBuffer:ref.buffer->object offset:GUARD+ref.offset atIndex:slot++];}
        [e setBytes:&a length:sizeof(a) atIndex:7];
        NSUInteger threads=std::min<NSUInteger>(256,pipe.maxTotalThreadsPerThreadgroup);
        [e dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];++dispatchCount;
    }
    void matrix(id<MTLComputeCommandEncoder> e,const char* name,std::initializer_list<Ref> refs,const Args& a,uint32_t rows,uint32_t cols){
        [e setComputePipelineState:pipelines.at(name)];uint32_t slot=0;
        for(auto ref:refs)[e setBuffer:ref.buffer->object offset:GUARD+ref.offset atIndex:slot++];
        [e setBytes:&a length:sizeof(a) atIndex:7];
        [e dispatchThreadgroups:MTLSizeMake((cols+15)/16,(rows+15)/16,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];++dispatchCount;
    }
    void linear(id<MTLComputeCommandEncoder> e,const char* x,P weight,P bias,const char* y,uint32_t m,uint32_t k,uint32_t n){
        Args a;a.m=m;a.k=k;a.n=n;matrix(e,weight==QKVW?"qkv_forward":"linear_forward",{r(x),p(weight),p(bias),r(y)},a,m,n);}
    void linearBack(id<MTLComputeCommandEncoder> e,const char* x,P weight,P bias,const char* dy,const char* dx,uint32_t m,uint32_t k,uint32_t n){
        Args a;a.m=m;a.k=k;a.n=n;
        matrix(e,weight==QKVW?"qkv_dx":"linear_dx",{r(dy),p(weight),r(dx)},a,m,k);
        matrix(e,"linear_dw",{r(x),r(dy),p(weight,true)},a,n,k);
        encode(e,"linear_db",{r(dy),p(bias,true)},a,n);}
    void norm(id<MTLComputeCommandEncoder> e,const char* x,P gamma,P beta,const char* y,int index,Args a){
        encode(e,"norm_forward",{r(x),p(gamma),p(beta),r(y),r("norm"+std::to_string(index)),r("inv"+std::to_string(index))},a,a.m);}
    void normBack(id<MTLComputeCommandEncoder> e,P gamma,P beta,const char* dy,const char* dx,int index,Args a){
        encode(e,"norm_dx",{r("norm"+std::to_string(index)),r("inv"+std::to_string(index)),p(gamma),r(dy),r(dx)},a,a.m);
        encode(e,"norm_dparam",{r("norm"+std::to_string(index)),r(dy),p(gamma,true),p(beta,true)},a,w);}
    void add(id<MTLComputeCommandEncoder> e,const char* x,const char* y,const char* z,uint32_t count){Args a;a.m=count;encode(e,"add",{r(x),r(y),r(z)},a,count);}
    void attention(id<MTLComputeCommandEncoder> e,Args a){
        encode(e,"attention_scores",{r("qkv"),r("scores"),r("score_lo")},a,a.b*4*a.t*a.t);
        encode(e,"attention_softmax",{r("scores"),r("score_lo"),r("probs")},a,a.b*4*a.t);
        encode(e,"attention_values",{r("qkv"),r("probs"),r("attention")},a,a.m*w);}
    void attentionBack(id<MTLComputeCommandEncoder> e,Args a){
        encode(e,"attention_da",{r("qkv"),r("dattention"),r("da"),r("da_lo")},a,a.b*4*a.t*a.t);
        encode(e,"attention_ds",{r("da"),r("da_lo"),r("probs"),r("ds"),r("ds_lo")},a,a.b*4*a.t);
        encode(e,"attention_dqkv",{r("qkv"),r("probs"),r("dattention"),r("ds"),r("ds_lo"),r("dqkv")},a,a.m*3*w);}
    void execute(uint64_t handle,uint32_t mode){
        auto batch=batches.at(handle);if(mode>3)throw std::runtime_error("Invalid execution mode");
        if(mode&& !batch.hasTargets)throw std::runtime_error("Targets required");
        Args a=args(batch.b,batch.t);uint32_t m=a.m;
        auto c=command();auto e=[c computeCommandEncoder];e.label=@"Transformer full operation";
        encode(e,"embedding",{{batch.tokens,0},p(TOKEN),p(POSITION),r("x")},a,m*w);
        norm(e,"x",LN1W,LN1B,"n1",1,a);linear(e,"n1",QKVW,QKVB,"qkv",m,w,3*w);attention(e,a);
        linear(e,"attention",ATTW,ATTB,"projected",m,w,w);add(e,"x","projected","r1",m*w);
        norm(e,"r1",LN2W,LN2B,"n2",2,a);linear(e,"n2",FF1W,FF1B,"ff1",m,w,2*w);
        Args count;count.m=m*2*w;encode(e,"relu_forward",{r("ff1"),r("relu")},count,count.m);
        linear(e,"relu",FF2W,FF2B,"ff2",m,2*w,w);add(e,"r1","ff2","r2",m*w);
        norm(e,"r2",LN3W,LN3B,"n3",3,a);linear(e,"n3",HEADW,HEADB,"logits",m,w,128);
        if(mode){encode(e,"cross_entropy",{r("logits"),{batch.targets,0},r("dlogits"),r("rowloss")},a,m);
            encode(e,"loss_mean",{r("rowloss"),r("loss")},a,1);}
        if(mode>=2){
            linearBack(e,"n3",HEADW,HEADB,"dlogits","dn3",m,w,128);
            normBack(e,LN3W,LN3B,"dn3","dr2",3,a);
            linearBack(e,"relu",FF2W,FF2B,"dr2","drelu",m,2*w,w);
            count.m=m*2*w;encode(e,"relu_backward",{r("ff1"),r("drelu"),r("dff1")},count,count.m);
            linearBack(e,"n2",FF1W,FF1B,"dff1","dn2",m,w,2*w);
            normBack(e,LN2W,LN2B,"dn2","temp",2,a);add(e,"dr2","temp","dr1",m*w);
            linearBack(e,"attention",ATTW,ATTB,"dr1","dattention",m,w,w);attentionBack(e,a);
            linearBack(e,"n1",QKVW,QKVB,"dqkv","dn1",m,w,3*w);
            normBack(e,LN1W,LN1B,"dn1","temp",1,a);add(e,"dr1","temp","dx",m*w);
            encode(e,"embedding_grad",{{batch.tokens,0},r("dx"),p(TOKEN,true),p(POSITION,true)},a,128*w);
        }
        if(mode==3){if(step==UINT32_MAX)throw std::runtime_error("Adam step counter exhausted");
            Args opt;opt.m=uint32_t(paramCount);opt.step=step+1;opt.epsilon=1e-8f;
            opt.corr1=float(1.0-std::pow(0.9,double(step+1)));opt.corr2=float(1.0-std::pow(0.999,double(step+1)));
            encode(e,"adam",{r("parameters"),r("gradients"),r("m"),r("v")},opt,paramCount);++step;}
        [e endEncoding];commit(c);lastB=batch.b;lastT=batch.t;lastMode=mode;
    }
    bool guards(){sync();for(auto& entry:buffers)if(!entry.second->guards())return false;
        for(auto& entry:inputs)if(!entry.second.values->guards())return false;
        for(auto& entry:batches)if(!entry.second.tokens->guards()||(entry.second.targets&&!entry.second.targets->guards()))return false;return true;}
    bool finite(const char* only,size_t count){
        auto c=command();auto e=[c computeCommandEncoder];Args a;encode(e,"finite_reset",{r("finite")},a,1);
        if(only){auto b=buffers.at(only);if(count*4>b->bytes)throw std::runtime_error("Finite scan out of range");
            a.m=uint32_t(count);encode(e,"finite_scan",{r(only),r("finite")},a,count);}
        else {for(auto name:{"parameters","m","v","gradients"}){a.m=uint32_t(paramCount);encode(e,"finite_scan",{r(name),r("finite")},a,paramCount);}
            if(lastB){a.m=lastB*lastT*128;encode(e,"finite_scan",{r("logits"),r("finite")},a,a.m);}
            if(lastMode){a.m=1;encode(e,"finite_scan",{r("loss"),r("finite")},a,1);}}
        [e endEncoding];commit(c);sync();return *static_cast<uint32_t*>(buffers.at("finite")->data())==0;
    }
    const char* capabilities(){
        NSMutableArray* families=[NSMutableArray array];
        for(int i=1;i<=10;++i)if([device supportsFamily:MTLGPUFamily(1000+i)])[families addObject:[NSString stringWithFormat:@"Apple%d",i]];
        for(int i=1;i<=2;++i)if([device supportsFamily:MTLGPUFamily(5000+i)])[families addObject:[NSString stringWithFormat:@"Metal%d",i+2]];
        NSMutableDictionary* pipes=[NSMutableDictionary dictionary];
        for(auto& [name,p]:pipelines)pipes[[NSString stringWithUTF8String:name.c_str()]]=@{@"thread_execution_width":@(p.threadExecutionWidth),
            @"max_threads_per_group":@(p.maxTotalThreadsPerThreadgroup),@"static_threadgroup_memory_bytes":@(p.staticThreadgroupMemoryLength)};
        NSDictionary* info=@{@"backend":@"handwritten-metal-v1",@"device":device.name,@"registry_id":@(device.registryID),
            @"unified_memory":@(device.hasUnifiedMemory),@"families":families,@"max_buffer_length":@(device.maxBufferLength),
            @"max_threadgroup_memory_bytes":@(device.maxThreadgroupMemoryLength),@"recommended_working_set_bytes":@(device.recommendedMaxWorkingSetSize),
            @"shader_language":@"MSL 3.1",@"compiler_math_mode":@"safe",@"floating_point_functions":@"precise",
            @"compiled_host_source_sha256":@TFM_HOST_SHA256,@"host_compiler":@__clang_version__,
            @"runtime_shader_compilation":@YES,@"framework_numerical_calls":@NO,@"native_fp64":@NO,
            @"attention_precision":@"FP32 two-float product/sum expansions through score centering and backward contractions",
            @"attention_scale_hi":@(attentionScaleHi),@"attention_scale_lo":@(attentionScaleLo),
            @"attention_scale_setup":@"Host double scalar 1/sqrt(head_width), split once into two FP32 constants; no tensor computation on CPU",
            @"normalization_precision":@"Two-float reductions, centered variance and backward contractions; FP32 saved norm/inverse/output",
            @"qkv_precision":@"Two-float forward and input-gradient contractions; other dense and QKV weight/bias gradients retain FP32 arithmetic",
            @"matrix_instruction_path":@"scalar FMA with 16x16 threadgroup tiles; no simdgroup matrix instruction",
            @"pipelines":pipes,@"adam_scalar_bookkeeping":@"Host step and double scalar pow for bias correction; all tensor arithmetic on GPU"};
        NSData* data=[NSJSONSerialization dataWithJSONObject:info options:NSJSONWritingSortedKeys error:nil];
        receipt.assign(static_cast<const char*>(data.bytes),data.length);return receipt.c_str();
    }
    const char* memory(){size_t allocated=0;for(auto& e:buffers)allocated+=e.second->object.length;
        for(auto& e:inputs)allocated+=e.second.values->object.length;
        for(auto& e:batches){allocated+=e.second.tokens->object.length;if(e.second.targets)allocated+=e.second.targets->object.length;}
        NSDictionary* info=@{@"owned_buffer_bytes_including_guards":@(allocated),@"device_current_allocated_bytes":@(device.currentAllocatedSize),
            @"dispatches":@(dispatchCount),@"command_buffers":@(commandCount),@"last_synchronized_gpu_seconds":@(lastGPUTime),
            @"definition":@"Owned shared MTLBuffer allocations plus guards; device counter includes pipeline/driver allocations, not peak RSS"};
        NSData* data=[NSJSONSerialization dataWithJSONObject:info options:NSJSONWritingSortedKeys error:nil];
        receipt.assign(static_cast<const char*>(data.bytes),data.length);return receipt.c_str();
    }
};

template<class F> int protect(F f){@autoreleasepool{try{f();lastError.clear();return 0;}
    catch(const std::exception& e){lastError=e.what();return -1;}}}
Context& ctx(void* p){if(!p)throw std::runtime_error("Null Metal backend");return *static_cast<Context*>(p);}
}

extern "C" {
const char* tfm_error(){return lastError.c_str();}
void* tfm_create(uint32_t w,uint32_t b,uint32_t t,const char* source,const float* initial,size_t n){
    Context* result=nullptr;protect([&]{result=new Context(w,b,t,source,initial,n);});return result;}
int tfm_destroy(void* p){return protect([&]{
    std::unique_ptr<Context> owned(static_cast<Context*>(p));
    if(owned)owned->sync();
});}
int tfm_sync(void* p){return protect([&]{ctx(p).sync();});}
const char* tfm_capabilities(void* p){const char* s=nullptr;protect([&]{s=ctx(p).capabilities();});return s;}
const char* tfm_memory(void* p){const char* s=nullptr;protect([&]{s=ctx(p).memory();});return s;}
int tfm_execute(void* p,uint64_t batch,uint32_t mode){return protect([&]{ctx(p).execute(batch,mode);});}
int tfm_read(void* p,const char* name,float* dest,size_t count){return protect([&]{auto& c=ctx(p);c.sync();auto b=c.buffers.at(name);
    if(!dest||count*4>b->bytes)throw std::runtime_error("Readback range mismatch");std::memcpy(dest,b->data(),count*4);});}
int tfm_write(void* p,const char* name,const float* src,size_t count){return protect([&]{auto& c=ctx(p);c.sync();auto b=c.buffers.at(name);
    if(!src||count*4!=b->bytes)throw std::runtime_error("State write range mismatch");
    for(size_t i=0;i<count;++i)if(!std::isfinite(src[i]))throw std::runtime_error("Nonfinite state");std::memcpy(b->data(),src,count*4);});}
int tfm_step(void* p,uint32_t* step,int write){return protect([&]{auto& c=ctx(p);c.sync();if(write)c.step=*step;else *step=c.step;});}
int tfm_guards(void* p){int result=-1;if(protect([&]{result=ctx(p).guards()?1:0;})<0)return -1;return result;}
int tfm_finite(void* p,const char* only,size_t count){int result=-1;if(protect([&]{result=ctx(p).finite(only,count)?1:0;})<0)return -1;return result;}
uint64_t tfm_batch(void* p,const int32_t* tokens,const int32_t* targets,uint32_t b,uint32_t t){uint64_t handle=0;
    protect([&]{auto& c=ctx(p);if(!tokens||b<1||b>c.maxB||t<1||t>c.maxT)throw std::runtime_error("Invalid batch shape");
        for(size_t i=0;i<b*t;++i)if(tokens[i]<0||tokens[i]>=128||(targets&&(targets[i]<0||targets[i]>=128)))throw std::runtime_error("Token outside [0,128)");
        Batch batch;batch.b=b;batch.t=t;batch.hasTargets=targets!=nullptr;batch.tokens=std::make_shared<Buffer>(c.device,b*t*4);
        std::memcpy(batch.tokens->data(),tokens,b*t*4);if(targets){batch.targets=std::make_shared<Buffer>(c.device,b*t*4);std::memcpy(batch.targets->data(),targets,b*t*4);}
        handle=c.nextHandle++;c.batches.emplace(handle,batch);});return handle;}
uint64_t tfm_upload(void* p,const int32_t* data,uint32_t rows,uint32_t t,uint32_t kind){uint64_t handle=0;
    protect([&]{auto& c=ctx(p);if(!data||rows<1||kind>1||rows>10000000)throw std::runtime_error("Invalid uploaded input");
        if(kind==0&&(t<1||t>c.maxT))throw std::runtime_error("Invalid dataset context");
        size_t n=kind?rows:size_t(rows)*(t+1);Input input;input.kind=kind;input.rows=rows;input.t=t;
        for(size_t i=0;i<n;++i)if(data[i]<0||(!kind&&data[i]>=128))throw std::runtime_error("Invalid token/index");
        input.values=std::make_shared<Buffer>(c.device,n*4);std::memcpy(input.values->data(),data,n*4);
        if(kind)input.indices.assign(data,data+n);handle=c.nextHandle++;c.inputs.emplace(handle,input);});return handle;}
uint64_t tfm_gather(void* p,uint64_t dataset,uint64_t schedule,uint32_t offset,uint32_t count){uint64_t handle=0;
    protect([&]{auto& c=ctx(p);auto& data=c.inputs.at(dataset);auto& ids=c.inputs.at(schedule);
        if(data.kind||ids.kind!=1||count<1||count>c.maxB||offset>ids.rows||count>ids.rows-offset)throw std::runtime_error("Invalid gather range");
        for(uint32_t i=0;i<count;++i)if(uint32_t(ids.indices[offset+i])>=data.rows)throw std::runtime_error("Gather index outside dataset");
        // Reuse a fixed staging batch per shape; queue ordering protects its lifetime.
        for(auto& item:c.batches)if(item.second.b==count&&item.second.t==data.t&&item.second.hasTargets&&item.first>>63){handle=item.first;break;}
        if(!handle){handle=(uint64_t(1)<<63)|c.nextHandle++;Batch batch;batch.b=count;batch.t=data.t;batch.hasTargets=true;
            batch.tokens=std::make_shared<Buffer>(c.device,count*data.t*4);batch.targets=std::make_shared<Buffer>(c.device,count*data.t*4);c.batches.emplace(handle,batch);}
        auto& batch=c.batches.at(handle);auto command=c.command();auto e=[command computeCommandEncoder];Args a=c.args(count,data.t);a.k=offset;
        c.encode(e,"gather_rows",{{data.values,0},{ids.values,0},{batch.tokens,0},{batch.targets,0}},a,count*data.t);[e endEncoding];c.commit(command);});return handle;}

// Primitive inspection calls allocate scratch only outside measured execution.
// They use the same cached pipelines as production and return named buffers.
int tfm_primitive(void* p,const char* kind,const float* const* data,const size_t* counts,uint32_t m,uint32_t n,uint32_t k,uint32_t b,uint32_t t,int backward){
    return protect([&]{auto& c=ctx(p);c.sync();std::string name(kind);Args a=c.args(b,t);a.m=m;a.n=n;a.k=k;
        auto input=[&](const char* label,int i,size_t size){if(!data[i]||counts[i]!=size)throw std::runtime_error("Primitive input shape mismatch");
            c.allocate(label,size);std::memcpy(c.buffers.at(label)->data(),data[i],size*4);};
        if(name=="attention"){
            if(b<1||b>c.maxB||t<1||t>c.maxT||m!=b*t)throw std::runtime_error("Primitive attention shape");
            size_t qcount=size_t(m)*3*c.w,dcount=size_t(m)*c.w;
            if(!data[0]||counts[0]!=qcount||(backward&&(!data[1]||counts[1]!=dcount)))throw std::runtime_error("Primitive attention input shape mismatch");
            std::memcpy(c.buffers.at("qkv")->data(),data[0],qcount*4);
            if(backward)std::memcpy(c.buffers.at("dattention")->data(),data[1],dcount*4);
            auto command=c.command();auto e=[command computeCommandEncoder];c.attention(e,a);if(backward)c.attentionBack(e,a);[e endEncoding];c.commit(command);
        }else if(name=="norm"){
            if(m<1||m>c.maxB*c.maxT||k!=c.w)throw std::runtime_error("Primitive norm shape");
            input("primitive_x",0,size_t(m)*k);input("primitive_gamma",1,k);input("primitive_beta",2,k);
            for(auto label:{"primitive_y","primitive_norm","primitive_dx"})c.allocate(label,size_t(m)*k);
            c.allocate("primitive_inv",m);c.allocate("primitive_dgamma",k);c.allocate("primitive_dbeta",k);
            if(backward)input("primitive_dy",3,size_t(m)*k);
            auto command=c.command();auto e=[command computeCommandEncoder];
            c.encode(e,"norm_forward",{c.r("primitive_x"),c.r("primitive_gamma"),c.r("primitive_beta"),c.r("primitive_y"),c.r("primitive_norm"),c.r("primitive_inv")},a,m);
            if(backward){c.encode(e,"norm_dx",{c.r("primitive_norm"),c.r("primitive_inv"),c.r("primitive_gamma"),c.r("primitive_dy"),c.r("primitive_dx")},a,m);
                c.encode(e,"norm_dparam",{c.r("primitive_norm"),c.r("primitive_dy"),c.r("primitive_dgamma"),c.r("primitive_dbeta")},a,k);}
            [e endEncoding];c.commit(command);
        }else if(name=="linear"){
            if(m<1||m>c.maxB*c.maxT||n<1||n>std::max(3*c.w,128u)||k<1||k>3*c.w)throw std::runtime_error("Primitive linear shape");
            input("primitive_x",0,size_t(m)*k);input("primitive_weight",1,size_t(n)*k);input("primitive_bias",2,n);
            c.allocate("primitive_y",size_t(m)*n);c.allocate("primitive_dx",size_t(m)*k);c.allocate("primitive_dw",size_t(n)*k);c.allocate("primitive_db",n);
            if(backward)input("primitive_dy",3,size_t(m)*n);
            auto command=c.command();auto e=[command computeCommandEncoder];
            bool qkv=n==3*c.w&&k==c.w;
            c.matrix(e,qkv?"qkv_forward":"linear_forward",{c.r("primitive_x"),c.r("primitive_weight"),c.r("primitive_bias"),c.r("primitive_y")},a,m,n);
            if(backward){c.matrix(e,qkv?"qkv_dx":"linear_dx",{c.r("primitive_dy"),c.r("primitive_weight"),c.r("primitive_dx")},a,m,k);
                c.matrix(e,"linear_dw",{c.r("primitive_x"),c.r("primitive_dy"),c.r("primitive_dw")},a,n,k);
                c.encode(e,"linear_db",{c.r("primitive_dy"),c.r("primitive_db")},a,n);}
            [e endEncoding];c.commit(command);
        }else throw std::runtime_error("Unknown primitive");
    });}
}
