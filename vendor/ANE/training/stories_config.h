// stories_config.h — Model config and structures (runtime-sized)
#pragma once
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#import <Accelerate/Accelerate.h>
#include <math.h>
#include <unistd.h>
#include <dispatch/dispatch.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

// ===== Runtime Model Configuration =====
// Dimensions are read from the model file header at startup.
// This replaces the old compile-time #defines (DIM=768, etc.)
typedef struct {
    int dim;           // embedding dimension (was 768)
    int hidden_dim;    // FFN hidden dimension (was 2048)
    int n_heads;       // number of attention heads (was 12)
    int head_dim;      // dim / n_heads (was 64)
    int seq_len;       // sequence length (was 256)
    int n_layers;      // number of transformer layers (was 12)
    int vocab_size;    // vocabulary size (was 32000)
    // Derived sizes (computed by model_config_init)
    int wq_sz, wo_sz;        // dim * dim
    int w1_sz, w2_sz, w3_sz; // hidden * dim / dim * hidden
    int score_ch;             // n_heads * seq_len
    int layer_params;         // total params per layer
    int total_params;         // total model params
} ModelConfig;

static ModelConfig g_mc;

// Validate model dimensions against sane upper bounds.
// Returns false if any dimension is out of range (prevents integer overflow,
// stack overflow from VLAs, and heap corruption from malicious model files).
static inline bool model_config_validate(const ModelConfig *mc) {
    if (mc->dim < 1 || mc->dim > 16384) return false;
    if (mc->hidden_dim < 1 || mc->hidden_dim > 65536) return false;
    if (mc->n_heads < 1 || mc->n_heads > 256) return false;
    if (mc->seq_len < 1 || mc->seq_len > 8192) return false;
    if (mc->n_layers < 1 || mc->n_layers > 256) return false;
    if (mc->vocab_size < 1 || mc->vocab_size > 200000) return false;
    if (mc->dim % mc->n_heads != 0) return false;  // head_dim must be integer
    return true;
}

static inline void model_config_init(ModelConfig *mc) {
    mc->head_dim = mc->dim / mc->n_heads;
    mc->wq_sz = mc->dim * mc->dim;
    mc->wo_sz = mc->dim * mc->dim;
    mc->w1_sz = mc->hidden_dim * mc->dim;
    mc->w2_sz = mc->dim * mc->hidden_dim;
    mc->w3_sz = mc->hidden_dim * mc->dim;
    mc->score_ch = mc->n_heads * mc->seq_len;
    mc->layer_params = 4 * mc->wq_sz + mc->w1_sz + mc->w2_sz + mc->w3_sz + 2 * mc->dim;
    mc->total_params = mc->n_layers * mc->layer_params + mc->dim + mc->vocab_size * mc->dim;
}

// Set g_mc to Stories110M defaults (used by tests and as fallback)
static inline void model_config_defaults(void) {
    g_mc.dim = 768; g_mc.hidden_dim = 2048; g_mc.n_heads = 12;
    g_mc.seq_len = 256; g_mc.n_layers = 12; g_mc.vocab_size = 32000;
    model_config_init(&g_mc);
}

// Backward-compatible macros — all code using DIM, HIDDEN, etc.
// now resolves to the runtime values in g_mc.
#define DIM       (g_mc.dim)
#define HIDDEN    (g_mc.hidden_dim)
#define HEADS     (g_mc.n_heads)
#define HD        (g_mc.head_dim)
#define SEQ       (g_mc.seq_len)
#define NLAYERS   (g_mc.n_layers)
#define VOCAB     (g_mc.vocab_size)
#define WQ_SZ     (g_mc.wq_sz)
#define WO_SZ     (g_mc.wo_sz)
#define W1_SZ     (g_mc.w1_sz)
#define W2_SZ     (g_mc.w2_sz)
#define W3_SZ     (g_mc.w3_sz)
#define SCORE_CH  (g_mc.score_ch)
#define LAYER_PARAMS (g_mc.layer_params)
#define TOTAL_PARAMS (g_mc.total_params)

// Non-model constants (these stay compile-time)
#define ACCUM_STEPS 10
#define MAX_COMPILES 100
#define KERNELS_PER_LAYER 5
#define TOTAL_WEIGHT_KERNELS (KERNELS_PER_LAYER * NLAYERS)

// Per-layer weight and optimizer state
typedef struct {
    float *Wq, *Wk, *Wv, *Wo;
    float *W1, *W2, *W3;
    float *rms_att, *rms_ffn;
} LayerWeights;

typedef struct {
    float *m, *v;
    size_t n;
} AdamState;

typedef struct {
    AdamState Wq, Wk, Wv, Wo;
    AdamState W1, W2, W3;
    AdamState rms_att, rms_ffn;
} LayerAdam;

// Per-layer activation buffers (saved for backward)
typedef struct {
    float *layer_in;    // [DIM, SEQ] input to this layer (for rmsnorm1 bwd)
    float *xnorm;      // [DIM, SEQ] rmsnorm1 output
    float *Q, *K, *V;  // [DIM, SEQ] QKV projections
    float *attn_out;    // [DIM, SEQ] attention output (before Wo)
    float *o_out;       // [DIM, SEQ] Wo output
    float *x2;          // [DIM, SEQ] residual after attn
    float *x2norm;      // [DIM, SEQ] rmsnorm2 output
    float *h1, *h3;     // [HIDDEN, SEQ] FFN intermediates
    float *silu_out;    // [HIDDEN, SEQ] SiLU(h1)*h3
    float *ffn_out;     // [DIM, SEQ] FFN output
} LayerActs;

// Per-layer gradient accumulators
typedef struct {
    float *Wq, *Wk, *Wv, *Wo;
    float *W1, *W2, *W3;
    float *rms_att, *rms_ffn;
} LayerGrads;

// ANE kernels per layer
typedef struct { void *model; IOSurfaceRef ioIn, ioOut; void *request; void *tmpDir; } Kern;
typedef struct {
    Kern *fwdAttn, *fwdFFN, *ffnBwd, *sdpaBwd1, *sdpaBwd2, *qkvBwd;
} LayerKernels;

// Checkpoint header
typedef struct {
    int magic;          // 0x424C5A54 "BLZT"
    int version;        // 2
    int step, total_steps;
    int n_layers, vocab_size, dim, hidden_dim, n_heads, seq_len;
    float lr, loss;
    double cum_compile, cum_train, cum_wall;
    int cum_steps, cum_batches;
    int adam_t;
    int pad[3];         // alignment
} CkptHdr;

// llama2.c model file header
typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
} Llama2Config;

// LoRA adapter: low-rank decomposition W_delta = B @ A * (alpha/rank)
typedef struct {
    float *A;    // [rank, in_dim] — initialized with kaiming uniform
    float *B;    // [out_dim, rank] — initialized to zero
    int rank, in_dim, out_dim;
} LoRAAdapter;

typedef struct {
    LoRAAdapter wo;  // Wo adapter (currently supported target)
} LayerLoRA;

typedef struct {
    AdamState A, B;
} LoRAAdam;

typedef struct {
    LoRAAdam wo;
} LayerLoRAAdam;

typedef struct {
    float *A, *B;
} LoRAGrad;

typedef struct {
    LoRAGrad wo;
} LayerLoRAGrad;

// Globals
static Class g_D, g_I, g_AR, g_AIO;
static mach_timebase_info_data_t g_tb;
static int g_compile_count = 0;

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
}
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

// Alloc helpers
static AdamState adam_alloc(size_t n) { AdamState s; s.m=(float*)calloc(n,4); s.v=(float*)calloc(n,4); s.n=n; return s; }
static void adam_free(AdamState *s) { free(s->m); free(s->v); }

static LayerWeights layer_weights_alloc(void) {
    LayerWeights w;
    w.Wq=(float*)malloc(WQ_SZ*4); w.Wk=(float*)malloc(WQ_SZ*4);
    w.Wv=(float*)malloc(WQ_SZ*4); w.Wo=(float*)malloc(WO_SZ*4);
    w.W1=(float*)malloc(W1_SZ*4); w.W2=(float*)malloc(W2_SZ*4); w.W3=(float*)malloc(W3_SZ*4);
    w.rms_att=(float*)malloc(DIM*4); w.rms_ffn=(float*)malloc(DIM*4);
    return w;
}
static void layer_weights_free(LayerWeights *w) {
    free(w->Wq);free(w->Wk);free(w->Wv);free(w->Wo);
    free(w->W1);free(w->W2);free(w->W3);
    free(w->rms_att);free(w->rms_ffn);
}
static LayerAdam layer_adam_alloc(void) {
    LayerAdam a;
    a.Wq=adam_alloc(WQ_SZ); a.Wk=adam_alloc(WQ_SZ); a.Wv=adam_alloc(WQ_SZ); a.Wo=adam_alloc(WO_SZ);
    a.W1=adam_alloc(W1_SZ); a.W2=adam_alloc(W2_SZ); a.W3=adam_alloc(W3_SZ);
    a.rms_att=adam_alloc(DIM); a.rms_ffn=adam_alloc(DIM);
    return a;
}
static void layer_adam_free(LayerAdam *a) {
    adam_free(&a->Wq);adam_free(&a->Wk);adam_free(&a->Wv);adam_free(&a->Wo);
    adam_free(&a->W1);adam_free(&a->W2);adam_free(&a->W3);
    adam_free(&a->rms_att);adam_free(&a->rms_ffn);
}
static LayerActs layer_acts_alloc(void) {
    LayerActs a;
    a.layer_in=(float*)malloc(SEQ*DIM*4);
    a.xnorm=(float*)malloc(SEQ*DIM*4); a.Q=(float*)malloc(SEQ*DIM*4);
    a.K=(float*)malloc(SEQ*DIM*4); a.V=(float*)malloc(SEQ*DIM*4);
    a.attn_out=(float*)malloc(SEQ*DIM*4); a.o_out=(float*)malloc(SEQ*DIM*4);
    a.x2=(float*)malloc(SEQ*DIM*4); a.x2norm=(float*)malloc(SEQ*DIM*4);
    a.h1=(float*)malloc(SEQ*HIDDEN*4); a.h3=(float*)malloc(SEQ*HIDDEN*4);
    a.silu_out=(float*)malloc(SEQ*HIDDEN*4); a.ffn_out=(float*)malloc(SEQ*DIM*4);
    return a;
}
static void layer_acts_free(LayerActs *a) {
    free(a->layer_in);free(a->xnorm);free(a->Q);free(a->K);free(a->V);
    free(a->attn_out);free(a->o_out);free(a->x2);free(a->x2norm);
    free(a->h1);free(a->h3);free(a->silu_out);free(a->ffn_out);
}
static LayerGrads layer_grads_alloc(void) {
    LayerGrads g;
    g.Wq=(float*)calloc(WQ_SZ,4); g.Wk=(float*)calloc(WQ_SZ,4);
    g.Wv=(float*)calloc(WQ_SZ,4); g.Wo=(float*)calloc(WO_SZ,4);
    g.W1=(float*)calloc(W1_SZ,4); g.W2=(float*)calloc(W2_SZ,4); g.W3=(float*)calloc(W3_SZ,4);
    g.rms_att=(float*)calloc(DIM,4); g.rms_ffn=(float*)calloc(DIM,4);
    return g;
}
static void layer_grads_zero(LayerGrads *g) {
    memset(g->Wq,0,WQ_SZ*4);memset(g->Wk,0,WQ_SZ*4);
    memset(g->Wv,0,WQ_SZ*4);memset(g->Wo,0,WO_SZ*4);
    memset(g->W1,0,W1_SZ*4);memset(g->W2,0,W2_SZ*4);memset(g->W3,0,W3_SZ*4);
    memset(g->rms_att,0,DIM*4);memset(g->rms_ffn,0,DIM*4);
}
static void layer_grads_free(LayerGrads *g) {
    free(g->Wq);free(g->Wk);free(g->Wv);free(g->Wo);
    free(g->W1);free(g->W2);free(g->W3);
    free(g->rms_att);free(g->rms_ffn);
}

// LoRA alloc/free helpers
static LoRAAdapter lora_adapter_alloc(int rank, int in_dim, int out_dim) {
    LoRAAdapter a;
    a.rank = rank; a.in_dim = in_dim; a.out_dim = out_dim;
    a.A = (float*)malloc((size_t)rank * in_dim * 4);
    a.B = (float*)calloc((size_t)out_dim * rank, 4);  // B starts at zero
    // Kaiming uniform init for A: U(-sqrt(1/rank), sqrt(1/rank))
    float scale = 1.0f / sqrtf((float)rank);
    for (size_t i = 0; i < (size_t)rank * in_dim; i++)
        a.A[i] = scale * (2.0f * (float)drand48() - 1.0f);
    return a;
}
static void lora_adapter_free(LoRAAdapter *a) { free(a->A); free(a->B); }

static LayerLoRA layer_lora_alloc(int rank, int lora_targets) {
    LayerLoRA l;
    memset(&l, 0, sizeof(l));
    if (lora_targets & 8) l.wo = lora_adapter_alloc(rank, DIM, DIM);
    return l;
}
static void layer_lora_free(LayerLoRA *l, int lora_targets) {
    if (lora_targets & 8) lora_adapter_free(&l->wo);
}

static LayerLoRAAdam layer_lora_adam_alloc(int rank, int lora_targets) {
    LayerLoRAAdam a;
    memset(&a, 0, sizeof(a));
    if (lora_targets & 8) {
        a.wo.A = adam_alloc((size_t)rank * DIM);
        a.wo.B = adam_alloc((size_t)DIM * rank);
    }
    return a;
}
static void layer_lora_adam_free(LayerLoRAAdam *a, int lora_targets) {
    if (lora_targets & 8) { adam_free(&a->wo.A); adam_free(&a->wo.B); }
}

static LayerLoRAGrad layer_lora_grad_alloc(int rank, int lora_targets) {
    LayerLoRAGrad g;
    memset(&g, 0, sizeof(g));
    if (lora_targets & 8) {
        g.wo.A = (float*)calloc((size_t)rank * DIM, 4);
        g.wo.B = (float*)calloc((size_t)DIM * rank, 4);
    }
    return g;
}
static void layer_lora_grad_zero(LayerLoRAGrad *g, int rank, int lora_targets) {
    if (lora_targets & 8) {
        memset(g->wo.A, 0, (size_t)rank * DIM * 4);
        memset(g->wo.B, 0, (size_t)DIM * rank * 4);
    }
}
static void layer_lora_grad_free(LayerLoRAGrad *g, int lora_targets) {
    if (lora_targets & 8) { free(g->wo.A); free(g->wo.B); }
}
