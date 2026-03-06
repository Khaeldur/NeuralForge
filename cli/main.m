// main.m — NeuralForge CLI entry point
// Commands: train, info, help
// All progress output is JSON on stdout; logs go to stderr
//
// Build: make neuralforge
// Usage: ./neuralforge train --model stories110M.bin --data tokens.bin --steps 1000
//        ./neuralforge info --model stories110M.bin
//        ./neuralforge train --resume --ckpt checkpoint.bin --data tokens.bin

#include "config.h"
#include "progress.h"
#include "tokenizer.h"

// Vendor headers — the full ANE training stack
#include "stories_io.h"
#include "stories_mil.h"
#include "stories_cpu_ops.h"
#include "ane_rmsnorm_bwd.h"
#include "ane_classifier.h"

#include <signal.h>

static volatile sig_atomic_t g_interrupted = 0;

static void sigint_handler(int sig) {
    (void)sig;
    g_interrupted = 1;
}

// ===== Weight loading from llama2.c format =====
static bool load_pretrained(LayerWeights *lw, float *rms_final, float *embed,
                            const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return false; }
    Llama2Config cfg;
    fread(&cfg, sizeof(cfg), 1, f);
    fprintf(stderr, "  Model: dim=%d hidden=%d layers=%d heads=%d vocab=%d seq=%d\n",
            cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads,
            abs(cfg.vocab_size), cfg.seq_len);
    if (cfg.dim != DIM || cfg.hidden_dim != HIDDEN || cfg.n_layers != NLAYERS) {
        fprintf(stderr, "  ERROR: Config mismatch!\n"); fclose(f); return false;
    }
    int V = abs(cfg.vocab_size);
    fread(embed, 4, V * DIM, f);
    for (int L=0; L<NLAYERS; L++) fread(lw[L].rms_att, 4, DIM, f);
    for (int L=0; L<NLAYERS; L++) fread(lw[L].Wq, 4, WQ_SZ, f);
    for (int L=0; L<NLAYERS; L++) fread(lw[L].Wk, 4, WQ_SZ, f);
    for (int L=0; L<NLAYERS; L++) fread(lw[L].Wv, 4, WQ_SZ, f);
    for (int L=0; L<NLAYERS; L++) fread(lw[L].Wo, 4, WO_SZ, f);
    for (int L=0; L<NLAYERS; L++) fread(lw[L].rms_ffn, 4, DIM, f);
    for (int L=0; L<NLAYERS; L++) fread(lw[L].W1, 4, W1_SZ, f);
    for (int L=0; L<NLAYERS; L++) fread(lw[L].W2, 4, W2_SZ, f);
    for (int L=0; L<NLAYERS; L++) fread(lw[L].W3, 4, W3_SZ, f);
    fread(rms_final, 4, DIM, f);
    fclose(f);
    fprintf(stderr, "  Loaded pretrained weights\n");
    return true;
}

// ===== Compile one layer's ANE kernels =====
static bool compile_layer_kernels(LayerKernels *lk, LayerWeights *w) {
    lk->fwdAttn = compile_kern_mil_w(gen_sdpa_fwd_taps(), (@{
        @"@model_path/weights/rms1.bin": @{@"offset":@0, @"data":build_blob(w->rms_att,1,DIM)},
        @"@model_path/weights/wq.bin":   @{@"offset":@0, @"data":build_blob(w->Wq,DIM,DIM)},
        @"@model_path/weights/wk.bin":   @{@"offset":@0, @"data":build_blob(w->Wk,DIM,DIM)},
        @"@model_path/weights/wv.bin":   @{@"offset":@0, @"data":build_blob(w->Wv,DIM,DIM)},
        @"@model_path/weights/wo.bin":   @{@"offset":@0, @"data":build_blob(w->Wo,DIM,DIM)},
        @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()},
    }), DIM*SEQ*2, 6*DIM*SEQ*2);
    lk->fwdFFN = compile_kern_mil_w(gen_ffn_fwd_taps(), (@{
        @"@model_path/weights/rms2.bin": @{@"offset":@0, @"data":build_blob(w->rms_ffn,1,DIM)},
        @"@model_path/weights/w1.bin":   @{@"offset":@0, @"data":build_blob(w->W1,HIDDEN,DIM)},
        @"@model_path/weights/w3.bin":   @{@"offset":@0, @"data":build_blob(w->W3,HIDDEN,DIM)},
        @"@model_path/weights/w2.bin":   @{@"offset":@0, @"data":build_blob(w->W2,DIM,HIDDEN)},
    }), DIM*SEQ*2, (2*DIM+3*HIDDEN)*SEQ*2);
    lk->ffnBwd = compile_kern_mil_w(gen_ffn_bwd(), (@{
        @"@model_path/weights/w2t.bin": @{@"offset":@0, @"data":build_blob_t(w->W2,DIM,HIDDEN)},
        @"@model_path/weights/w1t.bin": @{@"offset":@0, @"data":build_blob_t(w->W1,HIDDEN,DIM)},
        @"@model_path/weights/w3t.bin": @{@"offset":@0, @"data":build_blob_t(w->W3,HIDDEN,DIM)},
    }), (DIM+2*HIDDEN)*SEQ*2, (DIM+2*HIDDEN)*SEQ*2);
    lk->sdpaBwd1 = compile_kern_mil_w(gen_sdpa_bwd1(), (@{
        @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()},
        @"@model_path/weights/wot.bin":  @{@"offset":@0, @"data":build_blob_t(w->Wo,DIM,DIM)},
    }), 4*DIM*SEQ*2, (DIM+2*SCORE_CH)*SEQ*2);
    lk->qkvBwd = compile_kern_mil_w(gen_qkvb(), (@{
        @"@model_path/weights/wqt.bin": @{@"offset":@0, @"data":build_blob_t(w->Wq,DIM,DIM)},
        @"@model_path/weights/wkt.bin": @{@"offset":@0, @"data":build_blob_t(w->Wk,DIM,DIM)},
        @"@model_path/weights/wvt.bin": @{@"offset":@0, @"data":build_blob_t(w->Wv,DIM,DIM)},
    }), 3*DIM*SEQ*2, DIM*SEQ*2);
    return lk->fwdAttn && lk->fwdFFN && lk->ffnBwd && lk->sdpaBwd1 && lk->qkvBwd;
}

static Kern *compile_sdpa_bwd2(void) {
    return compile_kern_mil_w(gen_sdpa_bwd2(), @{}, (2*SCORE_CH+2*DIM)*SEQ*2, 2*DIM*SEQ*2);
}

static Kern *compile_rmsnorm_bwd_kern(const float *rms_w) {
    return compile_kern_mil_w(gen_rmsnorm_bwd(), (@{
        @"@model_path/weights/rms_w.bin": @{@"offset":@0, @"data":build_blob(rms_w, 1, DIM)},
    }), 2*DIM*SEQ*2, DIM*SEQ*2);
}

static Kern *compile_classifier_fwd(const float *embed) {
    return compile_kern_mil_w(gen_classifier_fwd(), (@{
        @"@model_path/weights/embed.bin": @{@"offset":@0, @"data":build_blob(embed, VOCAB, DIM)},
    }), DIM*SEQ*2, VOCAB*SEQ*2);
}

static Kern *compile_final_rmsnorm_kern(const float *rms_w) {
    return compile_kern_mil_w(gen_final_rmsnorm(), (@{
        @"@model_path/weights/rms_w.bin": @{@"offset":@0, @"data":build_blob(rms_w, 1, DIM)},
    }), DIM*SEQ*2, DIM*SEQ*2);
}

static Kern *compile_softmax_kern(void) {
    return compile_kern_mil_w(gen_softmax_vocab(), @{}, VOCAB*SEQ*2, VOCAB*SEQ*2);
}

static void free_layer_kernels(LayerKernels *lk) {
    free_kern(lk->fwdAttn); free_kern(lk->fwdFFN); free_kern(lk->ffnBwd);
    free_kern(lk->sdpaBwd1); free_kern(lk->qkvBwd);
    lk->fwdAttn = lk->fwdFFN = lk->ffnBwd = lk->sdpaBwd1 = lk->qkvBwd = NULL;
}

// ===== Checkpoint save/load =====
static void save_checkpoint(const char *path, int step, int total_steps, float lr,
                            float loss, double cc, double ct, double cw, int cs,
                            int cb, int adam_t, LayerWeights *lw, LayerAdam *la,
                            float *rms_final, AdamState *arms_final,
                            float *embed, AdamState *aembed) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        nf_emit_error("cannot write checkpoint", 8);
        fprintf(stderr, "ERROR: cannot open checkpoint for writing: %s\n", path);
        return;
    }
    CkptHdr h = {0};
    h.magic = 0x424C5A54; h.version = 2;
    h.step = step; h.total_steps = total_steps;
    h.n_layers = NLAYERS; h.vocab_size = VOCAB; h.dim = DIM;
    h.hidden_dim = HIDDEN; h.n_heads = HEADS; h.seq_len = SEQ;
    h.lr = lr; h.loss = loss;
    h.cum_compile = cc; h.cum_train = ct; h.cum_wall = cw;
    h.cum_steps = cs; h.cum_batches = cb; h.adam_t = adam_t;
    fwrite(&h, sizeof(h), 1, f);
    for (int L=0; L<NLAYERS; L++) {
        fwrite(lw[L].Wq,4,WQ_SZ,f); fwrite(lw[L].Wk,4,WQ_SZ,f);
        fwrite(lw[L].Wv,4,WQ_SZ,f); fwrite(lw[L].Wo,4,WO_SZ,f);
        fwrite(lw[L].W1,4,W1_SZ,f); fwrite(lw[L].W2,4,W2_SZ,f); fwrite(lw[L].W3,4,W3_SZ,f);
        fwrite(lw[L].rms_att,4,DIM,f); fwrite(lw[L].rms_ffn,4,DIM,f);
        fwrite(la[L].Wq.m,4,WQ_SZ,f); fwrite(la[L].Wq.v,4,WQ_SZ,f);
        fwrite(la[L].Wk.m,4,WQ_SZ,f); fwrite(la[L].Wk.v,4,WQ_SZ,f);
        fwrite(la[L].Wv.m,4,WQ_SZ,f); fwrite(la[L].Wv.v,4,WQ_SZ,f);
        fwrite(la[L].Wo.m,4,WO_SZ,f); fwrite(la[L].Wo.v,4,WO_SZ,f);
        fwrite(la[L].W1.m,4,W1_SZ,f); fwrite(la[L].W1.v,4,W1_SZ,f);
        fwrite(la[L].W2.m,4,W2_SZ,f); fwrite(la[L].W2.v,4,W2_SZ,f);
        fwrite(la[L].W3.m,4,W3_SZ,f); fwrite(la[L].W3.v,4,W3_SZ,f);
        fwrite(la[L].rms_att.m,4,DIM,f); fwrite(la[L].rms_att.v,4,DIM,f);
        fwrite(la[L].rms_ffn.m,4,DIM,f); fwrite(la[L].rms_ffn.v,4,DIM,f);
    }
    fwrite(rms_final,4,DIM,f);
    fwrite(arms_final->m,4,DIM,f); fwrite(arms_final->v,4,DIM,f);
    fwrite(embed,4,VOCAB*DIM,f);
    fwrite(aembed->m,4,VOCAB*DIM,f); fwrite(aembed->v,4,VOCAB*DIM,f);
    fclose(f);
}

static bool load_checkpoint(const char *path, int *step, int *total_steps,
                             float *lr, float *loss,
                             double *cc, double *ct, double *cw,
                             int *cs, int *cb, int *adam_t,
                             LayerWeights *lw, LayerAdam *la,
                             float *rms_final, AdamState *arms_final,
                             float *embed, AdamState *aembed) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    CkptHdr h;
    fread(&h, sizeof(h), 1, f);
    if (h.magic != 0x424C5A54 || h.version != 2) { fclose(f); return false; }
    *step = h.step; *total_steps = h.total_steps; *lr = h.lr; *loss = h.loss;
    *cc = h.cum_compile; *ct = h.cum_train; *cw = h.cum_wall;
    *cs = h.cum_steps; *cb = h.cum_batches; *adam_t = h.adam_t;
    for (int L=0; L<NLAYERS; L++) {
        fread(lw[L].Wq,4,WQ_SZ,f); fread(lw[L].Wk,4,WQ_SZ,f);
        fread(lw[L].Wv,4,WQ_SZ,f); fread(lw[L].Wo,4,WO_SZ,f);
        fread(lw[L].W1,4,W1_SZ,f); fread(lw[L].W2,4,W2_SZ,f); fread(lw[L].W3,4,W3_SZ,f);
        fread(lw[L].rms_att,4,DIM,f); fread(lw[L].rms_ffn,4,DIM,f);
        fread(la[L].Wq.m,4,WQ_SZ,f); fread(la[L].Wq.v,4,WQ_SZ,f);
        fread(la[L].Wk.m,4,WQ_SZ,f); fread(la[L].Wk.v,4,WQ_SZ,f);
        fread(la[L].Wv.m,4,WQ_SZ,f); fread(la[L].Wv.v,4,WQ_SZ,f);
        fread(la[L].Wo.m,4,WO_SZ,f); fread(la[L].Wo.v,4,WO_SZ,f);
        fread(la[L].W1.m,4,W1_SZ,f); fread(la[L].W1.v,4,W1_SZ,f);
        fread(la[L].W2.m,4,W2_SZ,f); fread(la[L].W2.v,4,W2_SZ,f);
        fread(la[L].W3.m,4,W3_SZ,f); fread(la[L].W3.v,4,W3_SZ,f);
        fread(la[L].rms_att.m,4,DIM,f); fread(la[L].rms_att.v,4,DIM,f);
        fread(la[L].rms_ffn.m,4,DIM,f); fread(la[L].rms_ffn.v,4,DIM,f);
    }
    fread(rms_final,4,DIM,f);
    fread(arms_final->m,4,DIM,f); fread(arms_final->v,4,DIM,f);
    fread(embed,4,VOCAB*DIM,f);
    fread(aembed->m,4,VOCAB*DIM,f); fread(aembed->v,4,VOCAB*DIM,f);
    fclose(f);
    return true;
}

// ===== Info command =====
static int nf_cmd_info(NFConfig *cfg) {
    FILE *f = fopen(cfg->model_path, "rb");
    if (!f) { nf_emit_error("cannot open model file", 1); return 1; }
    Llama2Config mcfg;
    fread(&mcfg, sizeof(mcfg), 1, f);
    fclose(f);
    fprintf(stdout, "{\"type\":\"info\",\"key\":\"model\",\"value\":{"
            "\"dim\":%d,\"hidden_dim\":%d,\"n_layers\":%d,\"n_heads\":%d,"
            "\"vocab_size\":%d,\"seq_len\":%d,"
            "\"params_millions\":%.1f}}\n",
            mcfg.dim, mcfg.hidden_dim, mcfg.n_layers, mcfg.n_heads,
            abs(mcfg.vocab_size), mcfg.seq_len,
            (float)TOTAL_PARAMS / 1e6);
    fflush(stdout);
    return 0;
}

// ===== Train command =====
// This is the core training loop extracted from train_large_ane.m
// with JSON progress output and SIGINT handling
static int nf_cmd_train(NFConfig *cfg, int argc, char **argv) {
    signal(SIGINT, sigint_handler);

    LayerWeights lw[NLAYERS]; LayerAdam la[NLAYERS];
    LayerActs acts[NLAYERS]; LayerGrads grads[NLAYERS]; LayerKernels kern[NLAYERS];
    for (int L=0; L<NLAYERS; L++) {
        lw[L] = layer_weights_alloc(); la[L] = layer_adam_alloc();
        acts[L] = layer_acts_alloc(); grads[L] = layer_grads_alloc();
        memset(&kern[L], 0, sizeof(LayerKernels));
    }
    float *rms_final = (float*)malloc(DIM*4);
    float *embed = (float*)malloc(VOCAB*DIM*4);
    float *grms_final = (float*)calloc(DIM, 4);
    float *gembed = (float*)calloc(VOCAB*DIM, 4);
    AdamState arms_final = adam_alloc(DIM);
    AdamState aembed = adam_alloc((size_t)VOCAB*DIM);
    double cum_compile=0, cum_train=0, cum_wall=0;
    int cum_steps=0, cum_batches=0;
    int adam_t = 0, start_step = 0;
    float last_loss = 999.0f, best_loss = 999.0f;
    int total_steps = cfg->total_steps;
    float lr = cfg->learning_rate;
    bool ane_extras = cfg->use_ane_extras;

    // Resume or load pretrained
    float resume_loss = 0;
    bool resuming = false;
    if (cfg->resume) {
        resuming = load_checkpoint(cfg->ckpt_path, &start_step, &total_steps, &lr,
                                   &resume_loss, &cum_compile, &cum_train, &cum_wall,
                                   &cum_steps, &cum_batches, &adam_t,
                                   lw, la, rms_final, &arms_final, embed, &aembed);
        if (resuming) {
            fprintf(stderr, "[RESUMED step %d, loss=%.4f]\n", start_step, resume_loss);
            last_loss = resume_loss;
            best_loss = resume_loss;
        }
    }
    if (!resuming) {
        if (!load_pretrained(lw, rms_final, embed, cfg->model_path)) {
            fprintf(stderr, "Pretrained load failed, using random init\n");
            srand48(cfg->seed);
            float scale_d=1.0f/sqrtf(DIM), scale_h=1.0f/sqrtf(HIDDEN);
            for (int L=0; L<NLAYERS; L++) {
                for(size_t i=0;i<WQ_SZ;i++){lw[L].Wq[i]=scale_d*(2*drand48()-1);lw[L].Wk[i]=scale_d*(2*drand48()-1);}
                for(size_t i=0;i<WQ_SZ;i++){lw[L].Wv[i]=scale_d*(2*drand48()-1);lw[L].Wo[i]=scale_d*(2*drand48()-1);}
                for(size_t i=0;i<W1_SZ;i++) lw[L].W1[i]=scale_h*(2*drand48()-1);
                for(size_t i=0;i<W2_SZ;i++) lw[L].W2[i]=scale_d*(2*drand48()-1);
                for(size_t i=0;i<W3_SZ;i++) lw[L].W3[i]=scale_h*(2*drand48()-1);
                for(int i=0;i<DIM;i++){lw[L].rms_att[i]=1.0f;lw[L].rms_ffn[i]=1.0f;}
            }
            for(int i=0;i<DIM;i++) rms_final[i]=1.0f;
            float escale=0.02f;
            for(size_t i=0;i<(size_t)VOCAB*DIM;i++) embed[i]=escale*(2*drand48()-1);
        }
    }

    // mmap token data
    int data_fd = open(cfg->data_path, O_RDONLY);
    if (data_fd < 0) {
        nf_emit_error("cannot open token data", 2);
        return 1;
    }
    struct stat st; fstat(data_fd, &st);
    size_t data_len = st.st_size;
    uint16_t *token_data = (uint16_t*)mmap(NULL, data_len, PROT_READ, MAP_PRIVATE, data_fd, 0);
    if (token_data == MAP_FAILED) { nf_emit_error("mmap failed", 3); return 1; }
    size_t n_tokens = data_len / 2;

    nf_emit_init(TOTAL_PARAMS, NLAYERS, DIM, HIDDEN, HEADS, SEQ, VOCAB);

    // Gradient buffers
    float *dy = (float*)malloc(SEQ*DIM*4);
    float *dffn = (float*)malloc(SEQ*DIM*4);
    float *dh1 = (float*)malloc(SEQ*HIDDEN*4);
    float *dh3 = (float*)malloc(SEQ*HIDDEN*4);
    float *dx_ffn = (float*)malloc(SEQ*DIM*4);
    float *dx2 = (float*)malloc(SEQ*DIM*4);
    float *do_out_buf = (float*)malloc(SEQ*DIM*4);
    float *dq = (float*)malloc(SEQ*DIM*4);
    float *dk = (float*)malloc(SEQ*DIM*4);
    float *dv = (float*)malloc(SEQ*DIM*4);
    float *dx_attn = (float*)malloc(SEQ*DIM*4);
    float *x_cur = (float*)malloc(SEQ*DIM*4);
    float *x_final = (float*)malloc(SEQ*DIM*4);
    float *logits = (float*)malloc(SEQ*VOCAB*4);
    float *dlogits = (float*)malloc(SEQ*VOCAB*4);
    float *probs = (float*)malloc(SEQ*VOCAB*4);

    // Compile static sdpaBwd2 kernels (no weights)
    Kern *sdpaBwd2[NLAYERS];
    for (int L=0; L<NLAYERS; L++) {
        sdpaBwd2[L] = compile_sdpa_bwd2();
        if (!sdpaBwd2[L]) { nf_emit_error("sdpaBwd2 compile failed", 4); return 1; }
    }

    Kern *rmsAttBwd[NLAYERS], *rmsFFNBwd[NLAYERS];
    memset(rmsAttBwd, 0, sizeof(rmsAttBwd));
    memset(rmsFFNBwd, 0, sizeof(rmsFFNBwd));

    Kern *softmaxKern = NULL;
    if (ane_extras) {
        softmaxKern = compile_softmax_kern();
        if (!softmaxKern) { nf_emit_error("softmax compile failed", 5); return 1; }
    }

    Kern *finalRmsKern = NULL, *classifierKern = NULL;

    dispatch_queue_t dw_q = dispatch_queue_create("nf.dw_cblas", DISPATCH_QUEUE_SERIAL);
    dispatch_group_t dw_grp = dispatch_group_create();

    double total_compile_ms=0, total_train_ms=0;
    int total_steps_done=0, total_batches=0;
    uint64_t t_wall_start = mach_absolute_time();
    srand48(cfg->seed + start_step);

    // FLOP constants for reporting
    double fwd_flops = NLAYERS * (4.0*2*DIM*DIM*SEQ + 2.0*2*DIM*HIDDEN*SEQ + 2.0*HIDDEN*DIM*SEQ);
    double sdpa_flops = NLAYERS * 2.0*HEADS*5*SEQ*SEQ*HD;
    double cls_flops = 2.0*VOCAB*DIM*SEQ;

    int step = start_step;
    while (step < total_steps && !g_interrupted) {
        // Check compile budget
        int kernels_needed = TOTAL_WEIGHT_KERNELS + (ane_extras ? 2*NLAYERS + 2 : 0);
        if (g_compile_count + kernels_needed > MAX_COMPILES) {
            for (int L=0; L<NLAYERS; L++) {
                free_layer_kernels(&kern[L]); free_kern(sdpaBwd2[L]);
                free_kern(rmsAttBwd[L]); free_kern(rmsFFNBwd[L]);
            }
            free_kern(softmaxKern); free_kern(finalRmsKern); free_kern(classifierKern);
            double wall = tb_ms(mach_absolute_time() - t_wall_start);
            save_checkpoint(cfg->ckpt_path, step, total_steps, lr, last_loss,
                total_compile_ms+cum_compile, total_train_ms+cum_train, wall+cum_wall,
                total_steps_done+cum_steps, total_batches+cum_batches, adam_t,
                lw, la, rms_final, &arms_final, embed, &aembed);

            nf_emit_restart(step, g_compile_count, "compile_budget_exceeded");

            // exec() restart — PID stays same, pipes stay open
            char steps_str[32]; snprintf(steps_str, sizeof(steps_str), "%d", total_steps);
            if (ane_extras)
                execl(argv[0], argv[0], "train", "--resume",
                      "--ckpt", cfg->ckpt_path, "--data", cfg->data_path,
                      "--steps", steps_str, NULL);
            else
                execl(argv[0], argv[0], "train", "--resume", "--no-ane-extras",
                      "--ckpt", cfg->ckpt_path, "--data", cfg->data_path,
                      "--steps", steps_str, NULL);
            nf_emit_error("exec restart failed", 99);
            return 1;
        }

        // Compile all layer kernels
        uint64_t tc = mach_absolute_time();
        for (int L=0; L<NLAYERS; L++) free_layer_kernels(&kern[L]);
        bool compile_ok = true;
        for (int L=0; L<NLAYERS; L++) {
            fprintf(stderr, "  Compiling layer %d/%d... (%d compiles)\r", L+1, NLAYERS, g_compile_count);
            if (!compile_layer_kernels(&kern[L], &lw[L])) {
                compile_ok = false; break;
            }
            if (ane_extras) {
                free_kern(rmsAttBwd[L]); free_kern(rmsFFNBwd[L]);
                rmsAttBwd[L] = compile_rmsnorm_bwd_kern(lw[L].rms_att);
                rmsFFNBwd[L] = compile_rmsnorm_bwd_kern(lw[L].rms_ffn);
                if (!rmsAttBwd[L] || !rmsFFNBwd[L]) { compile_ok = false; break; }
            }
        }
        if (!compile_ok) { g_compile_count = MAX_COMPILES; continue; }

        for (int L=0; L<NLAYERS; L++) {
            if (!sdpaBwd2[L]) {
                sdpaBwd2[L] = compile_sdpa_bwd2();
                if (!sdpaBwd2[L]) { nf_emit_error("sdpaBwd2 recompile failed", 6); return 1; }
            }
        }

        if (ane_extras) {
            free_kern(finalRmsKern); free_kern(classifierKern);
            finalRmsKern = compile_final_rmsnorm_kern(rms_final);
            classifierKern = compile_classifier_fwd(embed);
            if (!finalRmsKern || !classifierKern) { g_compile_count = MAX_COMPILES; continue; }
            if (!softmaxKern) {
                softmaxKern = compile_softmax_kern();
                if (!softmaxKern) { nf_emit_error("softmax recompile failed", 7); return 1; }
            }
        }

        double cms = tb_ms(mach_absolute_time() - tc);
        total_compile_ms += cms;
        fprintf(stderr, "  Compiled %d kernels in %.0fms                    \n", kernels_needed, cms);

        // Zero gradient accumulators
        for (int L=0; L<NLAYERS; L++) layer_grads_zero(&grads[L]);
        memset(grms_final, 0, DIM*4);
        memset(gembed, 0, (size_t)VOCAB*DIM*4);

        int steps_batch = 0;
        uint64_t tt = mach_absolute_time();

        for (int a=0; a<cfg->accum_steps && step<total_steps && !g_interrupted; a++, step++) {
            size_t max_pos = n_tokens - SEQ - 1;
            size_t pos = (size_t)(drand48() * max_pos);
            uint16_t *input_tokens = token_data + pos;
            uint16_t *target_tokens = token_data + pos + 1;

            // Embedding lookup
            embed_lookup(x_cur, embed, input_tokens, DIM, SEQ);

            // ===== FORWARD (12 layers) =====
            for (int L=0; L<NLAYERS; L++) {
                LayerActs *ac = &acts[L];
                memcpy(ac->layer_in, x_cur, SEQ*DIM*4);

                dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);

                io_write_fp16(kern[L].fwdAttn->ioIn, x_cur, DIM, SEQ);
                ane_eval(kern[L].fwdAttn);
                io_read_fp16(kern[L].fwdAttn->ioOut, ac->o_out, 0, DIM, SEQ);
                io_read_fp16(kern[L].fwdAttn->ioOut, ac->attn_out, 4*DIM, DIM, SEQ);
                io_read_fp16(kern[L].fwdAttn->ioOut, ac->xnorm, 5*DIM, DIM, SEQ);

                vDSP_vadd(x_cur, 1, ac->o_out, 1, ac->x2, 1, (vDSP_Length)(SEQ*DIM));

                io_write_fp16(kern[L].fwdFFN->ioIn, ac->x2, DIM, SEQ);
                ane_eval(kern[L].fwdFFN);
                io_read_fp16(kern[L].fwdFFN->ioOut, ac->ffn_out, 0, DIM, SEQ);
                io_read_fp16(kern[L].fwdFFN->ioOut, ac->h1, DIM, HIDDEN, SEQ);
                io_read_fp16(kern[L].fwdFFN->ioOut, ac->h3, DIM+HIDDEN, HIDDEN, SEQ);
                io_read_fp16(kern[L].fwdFFN->ioOut, ac->silu_out, DIM+2*HIDDEN, HIDDEN, SEQ);
                io_read_fp16(kern[L].fwdFFN->ioOut, ac->x2norm, DIM+3*HIDDEN, DIM, SEQ);

                vDSP_vadd(ac->x2, 1, ac->ffn_out, 1, x_cur, 1, (vDSP_Length)(SEQ*DIM));
            }

            if (ane_extras) {
                io_write_fp16(finalRmsKern->ioIn, x_cur, DIM, SEQ);
                ane_eval(finalRmsKern);
                io_read_fp16(finalRmsKern->ioOut, x_final, 0, DIM, SEQ);

                io_write_fp16(classifierKern->ioIn, x_final, DIM, SEQ);
                ane_eval(classifierKern);

                io_copy(softmaxKern->ioIn, 0, classifierKern->ioOut, 0, VOCAB, SEQ);
                ane_eval(softmaxKern);
                io_read_fp16(softmaxKern->ioOut, probs, 0, VOCAB, SEQ);
            } else {
                rmsnorm(x_final, x_cur, rms_final, DIM, SEQ);
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            VOCAB, SEQ, DIM, 1.0f, embed, DIM, x_final, SEQ, 0.0f, probs, SEQ);
                for (int t=0; t<SEQ; t++) {
                    float maxv = -1e30f;
                    for (int v=0; v<VOCAB; v++) { float val=probs[v*SEQ+t]; if(val>maxv) maxv=val; }
                    float sum = 0;
                    for (int v=0; v<VOCAB; v++) { probs[v*SEQ+t]=expf(probs[v*SEQ+t]-maxv); sum+=probs[v*SEQ+t]; }
                    for (int v=0; v<VOCAB; v++) probs[v*SEQ+t]/=sum;
                }
            }

            // NLL loss + gradient
            float total_loss = 0;
            float invS = 1.0f / SEQ;
            memcpy(dlogits, probs, (size_t)VOCAB*SEQ*4);
            for (int t=0; t<SEQ; t++) {
                int tgt = target_tokens[t];
                total_loss -= logf(probs[tgt*SEQ+t] + 1e-10f);
                dlogits[tgt*SEQ+t] -= 1.0f;
            }
            vDSP_vsmul(dlogits, 1, &invS, dlogits, 1, (vDSP_Length)((size_t)VOCAB*SEQ));
            float loss = total_loss / SEQ;
            last_loss = loss;
            if (loss < best_loss) best_loss = loss;

            // ===== BACKWARD =====
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        DIM, SEQ, VOCAB, 1.0f, embed, DIM, dlogits, SEQ, 0.0f, dy, SEQ);
            dispatch_group_async(dw_grp, dw_q, ^{
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            VOCAB, DIM, SEQ, 1.0f, dlogits, SEQ, x_final, SEQ, 1.0f, gembed, DIM);
            });

            {
                float *dx_rms_final = (float*)calloc(SEQ*DIM, 4);
                rmsnorm_bwd(dx_rms_final, grms_final, dy, x_cur, rms_final, DIM, SEQ);
                memcpy(dy, dx_rms_final, SEQ*DIM*4);
                free(dx_rms_final);
            }

            for (int L=NLAYERS-1; L>=0; L--) {
                LayerActs *ac = &acts[L];
                LayerGrads *gr = &grads[L];
                memcpy(dffn, dy, SEQ*DIM*4);

                io_write_fp16_at(kern[L].ffnBwd->ioIn, 0, dffn, DIM, SEQ);
                io_copy(kern[L].ffnBwd->ioIn, DIM, kern[L].fwdFFN->ioOut, DIM, 2*HIDDEN, SEQ);
                ane_eval(kern[L].ffnBwd);
                io_read_fp16(kern[L].ffnBwd->ioOut, dx_ffn, 0, DIM, SEQ);
                io_read_fp16(kern[L].ffnBwd->ioOut, dh1, DIM, HIDDEN, SEQ);
                io_read_fp16(kern[L].ffnBwd->ioOut, dh3, DIM+HIDDEN, HIDDEN, SEQ);

                float *capt_dffn = (float*)malloc(SEQ*DIM*4); memcpy(capt_dffn, dffn, SEQ*DIM*4);
                float *capt_silu = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_silu, ac->silu_out, SEQ*HIDDEN*4);
                float *capt_dh1 = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_dh1, dh1, SEQ*HIDDEN*4);
                float *capt_dh3 = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_dh3, dh3, SEQ*HIDDEN*4);
                float *capt_x2n = (float*)malloc(SEQ*DIM*4); memcpy(capt_x2n, ac->x2norm, SEQ*DIM*4);
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, HIDDEN, SEQ,
                                1.0f, capt_dffn, SEQ, capt_silu, SEQ, 1.0f, gr->W2, HIDDEN);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                1.0f, capt_dh1, SEQ, capt_x2n, SEQ, 1.0f, gr->W1, DIM);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                1.0f, capt_dh3, SEQ, capt_x2n, SEQ, 1.0f, gr->W3, DIM);
                    free(capt_dffn); free(capt_silu); free(capt_dh1); free(capt_dh3); free(capt_x2n);
                });

                if (ane_extras) {
                    io_write_fp16_at(rmsFFNBwd[L]->ioIn, 0, dx_ffn, DIM, SEQ);
                    io_write_fp16_at(rmsFFNBwd[L]->ioIn, DIM, ac->x2, DIM, SEQ);
                    ane_eval(rmsFFNBwd[L]);
                    io_read_fp16(rmsFFNBwd[L]->ioOut, dx2, 0, DIM, SEQ);
                }
                {
                    float *dw_tmp = (float*)calloc(DIM, 4);
                    float *dx_scratch = (float*)malloc(SEQ*DIM*4);
                    rmsnorm_bwd(dx_scratch, dw_tmp, dx_ffn, ac->x2, lw[L].rms_ffn, DIM, SEQ);
                    if (!ane_extras) memcpy(dx2, dx_scratch, SEQ*DIM*4);
                    for(int i=0;i<DIM;i++) gr->rms_ffn[i] += dw_tmp[i];
                    free(dx_scratch); free(dw_tmp);
                }
                for(int i=0;i<SEQ*DIM;i++) dx2[i] += dy[i];

                memcpy(do_out_buf, dx2, SEQ*DIM*4);
                float *capt_do = (float*)malloc(SEQ*DIM*4); memcpy(capt_do, do_out_buf, SEQ*DIM*4);
                float *capt_attn = (float*)malloc(SEQ*DIM*4); memcpy(capt_attn, ac->attn_out, SEQ*DIM*4);
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                1.0f, capt_do, SEQ, capt_attn, SEQ, 1.0f, gr->Wo, DIM);
                    free(capt_do); free(capt_attn);
                });

                io_copy(kern[L].sdpaBwd1->ioIn, 0, kern[L].fwdAttn->ioOut, DIM, 3*DIM, SEQ);
                io_write_fp16_at(kern[L].sdpaBwd1->ioIn, 3*DIM, dx2, DIM, SEQ);
                ane_eval(kern[L].sdpaBwd1);
                io_copy(sdpaBwd2[L]->ioIn, 0, kern[L].sdpaBwd1->ioOut, DIM, 2*SCORE_CH, SEQ);
                io_copy(sdpaBwd2[L]->ioIn, 2*SCORE_CH, kern[L].fwdAttn->ioOut, DIM, 2*DIM, SEQ);
                ane_eval(sdpaBwd2[L]);

                io_read_fp16(sdpaBwd2[L]->ioOut, dq, 0, DIM, SEQ);
                io_read_fp16(sdpaBwd2[L]->ioOut, dk, DIM, DIM, SEQ);
                io_read_fp16(kern[L].sdpaBwd1->ioOut, dv, 0, DIM, SEQ);

                float *capt_dq = (float*)malloc(SEQ*DIM*4); memcpy(capt_dq, dq, SEQ*DIM*4);
                float *capt_dk = (float*)malloc(SEQ*DIM*4); memcpy(capt_dk, dk, SEQ*DIM*4);
                float *capt_dv = (float*)malloc(SEQ*DIM*4); memcpy(capt_dv, dv, SEQ*DIM*4);
                float *capt_xn = (float*)malloc(SEQ*DIM*4); memcpy(capt_xn, ac->xnorm, SEQ*DIM*4);
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                1.0f, capt_dq, SEQ, capt_xn, SEQ, 1.0f, gr->Wq, DIM);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                1.0f, capt_dk, SEQ, capt_xn, SEQ, 1.0f, gr->Wk, DIM);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                1.0f, capt_dv, SEQ, capt_xn, SEQ, 1.0f, gr->Wv, DIM);
                    free(capt_dq); free(capt_dk); free(capt_dv); free(capt_xn);
                });

                io_copy(kern[L].qkvBwd->ioIn, 0, sdpaBwd2[L]->ioOut, 0, 2*DIM, SEQ);
                io_copy(kern[L].qkvBwd->ioIn, 2*DIM, kern[L].sdpaBwd1->ioOut, 0, DIM, SEQ);
                ane_eval(kern[L].qkvBwd);
                io_read_fp16(kern[L].qkvBwd->ioOut, dx_attn, 0, DIM, SEQ);

                float *dx_rms1 = (float*)malloc(SEQ*DIM*4);
                if (ane_extras) {
                    io_write_fp16_at(rmsAttBwd[L]->ioIn, 0, dx_attn, DIM, SEQ);
                    io_write_fp16_at(rmsAttBwd[L]->ioIn, DIM, ac->layer_in, DIM, SEQ);
                    ane_eval(rmsAttBwd[L]);
                    io_read_fp16(rmsAttBwd[L]->ioOut, dx_rms1, 0, DIM, SEQ);
                }
                {
                    float *dw_tmp = (float*)calloc(DIM, 4);
                    float *dx_scratch = (float*)malloc(SEQ*DIM*4);
                    rmsnorm_bwd(dx_scratch, dw_tmp, dx_attn, ac->layer_in, lw[L].rms_att, DIM, SEQ);
                    if (!ane_extras) memcpy(dx_rms1, dx_scratch, SEQ*DIM*4);
                    for(int i=0;i<DIM;i++) gr->rms_att[i] += dw_tmp[i];
                    free(dx_scratch); free(dw_tmp);
                }

                for(int i=0;i<SEQ*DIM;i++) dy[i] = dx_rms1[i] + dx2[i];
                free(dx_rms1);
            }

            dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
            embed_backward(gembed, dy, input_tokens, DIM, SEQ);
            steps_batch++;

            // Emit per-step progress
            double step_ms = tb_ms(mach_absolute_time() - tt) / steps_batch;
            double ane_tflops = (fwd_flops*2 + sdpa_flops + cls_flops) / (step_ms * 1e9);
            double total_tflops = (fwd_flops*3 + sdpa_flops + cls_flops*3) / (step_ms * 1e9);
            nf_emit_step(step, total_steps, loss, lr, step_ms, ane_tflops, total_tflops);
        }

        double tms = tb_ms(mach_absolute_time() - tt);
        total_train_ms += tms;
        total_steps_done += steps_batch;
        total_batches++;

        dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);

        // Adam update
        float gsc = 1.0f / steps_batch;
        adam_t++;
        for (int L=0; L<NLAYERS; L++) {
            LayerGrads *g = &grads[L];
            for(size_t i=0;i<WQ_SZ;i++){g->Wq[i]*=gsc;g->Wk[i]*=gsc;g->Wv[i]*=gsc;g->Wo[i]*=gsc;}
            for(size_t i=0;i<W1_SZ;i++) g->W1[i]*=gsc;
            for(size_t i=0;i<W2_SZ;i++) g->W2[i]*=gsc;
            for(size_t i=0;i<W3_SZ;i++) g->W3[i]*=gsc;
            for(int i=0;i<DIM;i++){g->rms_att[i]*=gsc; g->rms_ffn[i]*=gsc;}
            adam_update(lw[L].Wq, g->Wq, &la[L].Wq, adam_t, lr, cfg->beta1, cfg->beta2, cfg->eps);
            adam_update(lw[L].Wk, g->Wk, &la[L].Wk, adam_t, lr, cfg->beta1, cfg->beta2, cfg->eps);
            adam_update(lw[L].Wv, g->Wv, &la[L].Wv, adam_t, lr, cfg->beta1, cfg->beta2, cfg->eps);
            adam_update(lw[L].Wo, g->Wo, &la[L].Wo, adam_t, lr, cfg->beta1, cfg->beta2, cfg->eps);
            adam_update(lw[L].W1, g->W1, &la[L].W1, adam_t, lr, cfg->beta1, cfg->beta2, cfg->eps);
            adam_update(lw[L].W2, g->W2, &la[L].W2, adam_t, lr, cfg->beta1, cfg->beta2, cfg->eps);
            adam_update(lw[L].W3, g->W3, &la[L].W3, adam_t, lr, cfg->beta1, cfg->beta2, cfg->eps);
            adam_update(lw[L].rms_att, g->rms_att, &la[L].rms_att, adam_t, lr, cfg->beta1, cfg->beta2, cfg->eps);
            adam_update(lw[L].rms_ffn, g->rms_ffn, &la[L].rms_ffn, adam_t, lr, cfg->beta1, cfg->beta2, cfg->eps);
        }
        for(int i=0;i<DIM;i++) grms_final[i]*=gsc;
        adam_update(rms_final, grms_final, &arms_final, adam_t, lr, cfg->beta1, cfg->beta2, cfg->eps);
        for(size_t i=0;i<(size_t)VOCAB*DIM;i++) gembed[i]*=gsc;
        adam_update(embed, gembed, &aembed, adam_t, lr, cfg->beta1, cfg->beta2, cfg->eps);

        nf_emit_batch(total_batches, step, last_loss, best_loss, cms, tms, g_compile_count);

        // Periodic checkpoint
        if (step > 0 && step % cfg->checkpoint_every == 0) {
            double wall = tb_ms(mach_absolute_time() - t_wall_start);
            save_checkpoint(cfg->ckpt_path, step, total_steps, lr, last_loss,
                total_compile_ms+cum_compile, total_train_ms+cum_train, wall+cum_wall,
                total_steps_done+cum_steps, total_batches+cum_batches, adam_t,
                lw, la, rms_final, &arms_final, embed, &aembed);
            nf_emit_checkpoint(cfg->ckpt_path, step, last_loss);
        }
    }

    // Handle SIGINT graceful shutdown
    if (g_interrupted) {
        double wall = tb_ms(mach_absolute_time() - t_wall_start);
        save_checkpoint(cfg->ckpt_path, step, total_steps, lr, last_loss,
            total_compile_ms+cum_compile, total_train_ms+cum_train, wall+cum_wall,
            total_steps_done+cum_steps, total_batches+cum_batches, adam_t,
            lw, la, rms_final, &arms_final, embed, &aembed);
        nf_emit_checkpoint(cfg->ckpt_path, step, last_loss);
        fprintf(stderr, "\n[Interrupted — checkpoint saved at step %d]\n", step);
    }

    // Final report
    double wall = tb_ms(mach_absolute_time() - t_wall_start);
    total_compile_ms += cum_compile; total_train_ms += cum_train;
    wall += cum_wall; total_steps_done += cum_steps;

    double ane_flops_total = (fwd_flops*2 + sdpa_flops + cls_flops) * total_steps_done;
    double all_flops_total = (fwd_flops*3 + sdpa_flops + cls_flops*3) * total_steps_done;
    double ane_tflops = ane_flops_total / (total_train_ms * 1e9);
    double total_tflops = all_flops_total / (total_train_ms * 1e9);

    nf_emit_done(total_steps_done, last_loss, wall/1000.0, ane_tflops, total_tflops);

    // Cleanup
    for (int L=0; L<NLAYERS; L++) {
        free_layer_kernels(&kern[L]); free_kern(sdpaBwd2[L]);
        free_kern(rmsAttBwd[L]); free_kern(rmsFFNBwd[L]);
        layer_weights_free(&lw[L]); layer_adam_free(&la[L]);
        layer_acts_free(&acts[L]); layer_grads_free(&grads[L]);
    }
    free_kern(softmaxKern); free_kern(finalRmsKern); free_kern(classifierKern);
    munmap(token_data, data_len); close(data_fd);
    free(rms_final); free(embed); free(grms_final); free(gembed);
    adam_free(&arms_final); adam_free(&aembed);
    free(dy); free(dffn); free(dh1); free(dh3); free(dx_ffn); free(dx2);
    free(do_out_buf); free(dq); free(dk); free(dv); free(dx_attn);
    free(x_cur); free(x_final); free(logits); free(dlogits); free(probs);

    return 0;
}

// ===== Tokenize command =====
static int nf_cmd_tokenize(int argc, char **argv) {
    const char *input_path = NULL;
    const char *output_path = "tokens.bin";
    const char *tokenizer_path = "tokenizer.bin";

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i+1 < argc) input_path = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i+1 < argc) output_path = argv[++i];
        else if (strcmp(argv[i], "--tokenizer") == 0 && i+1 < argc) tokenizer_path = argv[++i];
    }

    if (!input_path) {
        nf_emit_error("--input path required", 10);
        return 1;
    }

    NFTokenizer tok;
    if (!nf_tokenizer_load(&tok, tokenizer_path, VOCAB)) {
        nf_emit_error("cannot load tokenizer", 11);
        return 1;
    }
    fprintf(stderr, "Loaded tokenizer: %d vocab, max_len=%d\n", tok.vocab_size, tok.max_token_length);

    // Read input text file
    FILE *fin = fopen(input_path, "r");
    if (!fin) { nf_emit_error("cannot open input file", 12); return 1; }
    fseek(fin, 0, SEEK_END);
    long fsize = ftell(fin);
    if (fsize <= 0 || fsize > (long)2e9) {
        nf_emit_error("input file empty or too large (>2GB)", 14);
        fclose(fin); return 1;
    }
    fseek(fin, 0, SEEK_SET);
    char *text = (char *)malloc(fsize + 1);
    if (!text) { nf_emit_error("out of memory for text", 15); fclose(fin); return 1; }
    size_t nread = fread(text, 1, fsize, fin);
    text[nread] = '\0';
    fclose(fin);
    fprintf(stderr, "Read %ld bytes of text\n", fsize);

    // Tokenize in chunks (SEQ-sized windows)
    size_t max_tokens = (size_t)fsize;  // Upper bound: 1 token per byte
    uint16_t *tokens = (uint16_t *)malloc(max_tokens * sizeof(uint16_t));
    if (!tokens) { nf_emit_error("out of memory for tokens", 16); free(text); return 1; }

    int total_tokens = nf_tokenizer_encode(&tok, text, tokens, (int)max_tokens);
    fprintf(stderr, "Encoded %d tokens (%.1f bytes/token)\n",
            total_tokens, (float)fsize / total_tokens);

    // Write output
    FILE *fout = fopen(output_path, "wb");
    if (!fout) { nf_emit_error("cannot open output file", 13); return 1; }
    fwrite(tokens, sizeof(uint16_t), total_tokens, fout);
    fclose(fout);

    char esc_in[PATH_MAX*2], esc_out[PATH_MAX*2];
    nf_json_escape(input_path, esc_in, sizeof(esc_in));
    nf_json_escape(output_path, esc_out, sizeof(esc_out));
    fprintf(stdout, "{\"type\":\"tokenize\",\"input\":\"%s\",\"output\":\"%s\","
            "\"tokens\":%d,\"bytes\":%ld}\n", esc_in, esc_out, total_tokens, fsize);
    fflush(stdout);

    free(text);
    free(tokens);
    nf_tokenizer_free(&tok);
    return 0;
}

// ===== Export command =====
static int nf_cmd_export(int argc, char **argv) {
    const char *ckpt_path = NULL;
    const char *output_path = NULL;
    const char *format = "llama2c";

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--ckpt") == 0 && i+1 < argc) ckpt_path = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i+1 < argc) output_path = argv[++i];
        else if (strcmp(argv[i], "--format") == 0 && i+1 < argc) format = argv[++i];
    }

    if (!ckpt_path || !output_path) {
        nf_emit_error("--ckpt and --output required", 20);
        return 1;
    }

    // Load checkpoint
    LayerWeights lw[NLAYERS]; LayerAdam la[NLAYERS];
    for (int L=0; L<NLAYERS; L++) { lw[L] = layer_weights_alloc(); la[L] = layer_adam_alloc(); }
    float *rms_final = (float*)malloc(DIM*4);
    float *embed = (float*)malloc(VOCAB*DIM*4);
    AdamState arms_final = adam_alloc(DIM);
    AdamState aembed = adam_alloc((size_t)VOCAB*DIM);

    int step, total_steps, cum_steps, cum_batches, adam_t;
    float lr, loss;
    double cum_compile, cum_train, cum_wall;

    if (!load_checkpoint(ckpt_path, &step, &total_steps, &lr, &loss,
                          &cum_compile, &cum_train, &cum_wall,
                          &cum_steps, &cum_batches, &adam_t,
                          lw, la, rms_final, &arms_final, embed, &aembed)) {
        nf_emit_error("cannot load checkpoint", 21);
        return 1;
    }
    fprintf(stderr, "Loaded checkpoint: step=%d, loss=%.4f\n", step, loss);

    if (strcmp(format, "llama2c") == 0) {
        // Export as llama2.c format: config header + raw weights
        FILE *fout = fopen(output_path, "wb");
        if (!fout) { nf_emit_error("cannot open output", 22); return 1; }

        Llama2Config cfg;
        cfg.dim = DIM; cfg.hidden_dim = HIDDEN; cfg.n_layers = NLAYERS;
        cfg.n_heads = HEADS; cfg.n_kv_heads = HEADS;
        cfg.vocab_size = -VOCAB;  // Negative = shared embedding
        cfg.seq_len = SEQ;
        fwrite(&cfg, sizeof(cfg), 1, fout);

        // Weights in llama2.c order
        fwrite(embed, 4, VOCAB*DIM, fout);
        for (int L=0; L<NLAYERS; L++) fwrite(lw[L].rms_att, 4, DIM, fout);
        for (int L=0; L<NLAYERS; L++) fwrite(lw[L].Wq, 4, WQ_SZ, fout);
        for (int L=0; L<NLAYERS; L++) fwrite(lw[L].Wk, 4, WQ_SZ, fout);
        for (int L=0; L<NLAYERS; L++) fwrite(lw[L].Wv, 4, WQ_SZ, fout);
        for (int L=0; L<NLAYERS; L++) fwrite(lw[L].Wo, 4, WO_SZ, fout);
        for (int L=0; L<NLAYERS; L++) fwrite(lw[L].rms_ffn, 4, DIM, fout);
        for (int L=0; L<NLAYERS; L++) fwrite(lw[L].W1, 4, W1_SZ, fout);
        for (int L=0; L<NLAYERS; L++) fwrite(lw[L].W2, 4, W2_SZ, fout);
        for (int L=0; L<NLAYERS; L++) fwrite(lw[L].W3, 4, W3_SZ, fout);
        fwrite(rms_final, 4, DIM, fout);

        long bytes = ftell(fout);
        fclose(fout);

        char esc_path[PATH_MAX*2];
        nf_json_escape(output_path, esc_path, sizeof(esc_path));
        fprintf(stdout, "{\"type\":\"export\",\"format\":\"llama2c\",\"output\":\"%s\","
                "\"bytes\":%ld,\"step\":%d,\"loss\":%.6f}\n",
                esc_path, bytes, step, loss);
        fflush(stdout);
    } else if (strcmp(format, "gguf") == 0) {
        // Full GGUF v3 export with tensor data
        FILE *fout = fopen(output_path, "wb");
        if (!fout) { nf_emit_error("cannot open output", 22); return 1; }

        #define GGUF_ALIGN 32
        #define GGML_F32 0

        // Helper macros
        #define GGUF_STR(s) do { uint64_t l = strlen(s); fwrite(&l, 8, 1, fout); fwrite(s, 1, l, fout); } while(0)
        #define GGUF_KV_U32(key, val) do { GGUF_STR(key); uint32_t t=4; fwrite(&t,4,1,fout); uint32_t v=val; fwrite(&v,4,1,fout); } while(0)
        #define GGUF_KV_STR(key, val) do { GGUF_STR(key); uint32_t t=8; fwrite(&t,4,1,fout); GGUF_STR(val); } while(0)
        #define GGUF_KV_F32(key, val) do { GGUF_STR(key); uint32_t t=6; fwrite(&t,4,1,fout); float v=val; fwrite(&v,4,1,fout); } while(0)

        // Build tensor list: name, data pointer, dims, sizes
        typedef struct { const char *name; float *data; int n_dims; uint64_t dims[2]; size_t nbytes; } TensorInfo;

        int n_t = 0;
        TensorInfo tensors[256];  // Max tensors

        // Token embedding
        tensors[n_t++] = (TensorInfo){"token_embd.weight", embed, 2, {DIM, VOCAB}, (size_t)VOCAB*DIM*4};

        for (int L = 0; L < NLAYERS; L++) {
            char name[64];
            snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", L);
            tensors[n_t++] = (TensorInfo){strdup(name), lw[L].rms_att, 1, {DIM, 0}, DIM*4};
            snprintf(name, sizeof(name), "blk.%d.attn_q.weight", L);
            tensors[n_t++] = (TensorInfo){strdup(name), lw[L].Wq, 2, {DIM, DIM}, (size_t)DIM*DIM*4};
            snprintf(name, sizeof(name), "blk.%d.attn_k.weight", L);
            tensors[n_t++] = (TensorInfo){strdup(name), lw[L].Wk, 2, {DIM, DIM}, (size_t)DIM*DIM*4};
            snprintf(name, sizeof(name), "blk.%d.attn_v.weight", L);
            tensors[n_t++] = (TensorInfo){strdup(name), lw[L].Wv, 2, {DIM, DIM}, (size_t)DIM*DIM*4};
            snprintf(name, sizeof(name), "blk.%d.attn_output.weight", L);
            tensors[n_t++] = (TensorInfo){strdup(name), lw[L].Wo, 2, {DIM, DIM}, (size_t)DIM*DIM*4};
            snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", L);
            tensors[n_t++] = (TensorInfo){strdup(name), lw[L].rms_ffn, 1, {DIM, 0}, DIM*4};
            snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", L);
            tensors[n_t++] = (TensorInfo){strdup(name), lw[L].W1, 2, {DIM, HIDDEN}, (size_t)HIDDEN*DIM*4};
            snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", L);
            tensors[n_t++] = (TensorInfo){strdup(name), lw[L].W2, 2, {HIDDEN, DIM}, (size_t)DIM*HIDDEN*4};
            snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", L);
            tensors[n_t++] = (TensorInfo){strdup(name), lw[L].W3, 2, {DIM, HIDDEN}, (size_t)HIDDEN*DIM*4};
        }

        // Final norm
        tensors[n_t++] = (TensorInfo){"output_norm.weight", rms_final, 1, {DIM, 0}, DIM*4};

        // GGUF header
        uint32_t magic = 0x46475547;
        uint32_t version = 3;
        fwrite(&magic, 4, 1, fout);
        fwrite(&version, 4, 1, fout);

        uint64_t n_tensors_u64 = n_t;
        uint64_t n_kv = 10;
        fwrite(&n_tensors_u64, 8, 1, fout);
        fwrite(&n_kv, 8, 1, fout);

        // KV metadata
        GGUF_KV_STR("general.architecture", "llama");
        GGUF_KV_STR("general.name", "NeuralForge Stories 110M");
        GGUF_KV_U32("llama.embedding_length", DIM);
        GGUF_KV_U32("llama.feed_forward_length", HIDDEN);
        GGUF_KV_U32("llama.block_count", NLAYERS);
        GGUF_KV_U32("llama.attention.head_count", HEADS);
        GGUF_KV_U32("llama.attention.head_count_kv", HEADS);
        GGUF_KV_U32("llama.context_length", SEQ);
        GGUF_KV_F32("llama.attention.layer_norm_rms_epsilon", 1e-5f);
        GGUF_KV_STR("tokenizer.ggml.model", "llama");

        // Tensor info headers (compute offsets into data block)
        uint64_t data_offset = 0;
        for (int i = 0; i < n_t; i++) {
            // Align offset
            uint64_t pad = (GGUF_ALIGN - (data_offset % GGUF_ALIGN)) % GGUF_ALIGN;
            data_offset += pad;

            // Name
            GGUF_STR(tensors[i].name);

            // Dimensions
            uint32_t ndims = tensors[i].n_dims;
            fwrite(&ndims, 4, 1, fout);
            for (int d = 0; d < tensors[i].n_dims; d++) {
                fwrite(&tensors[i].dims[d], 8, 1, fout);
            }

            // Type (F32)
            uint32_t dtype = GGML_F32;
            fwrite(&dtype, 4, 1, fout);

            // Offset
            fwrite(&data_offset, 8, 1, fout);

            data_offset += tensors[i].nbytes;
        }

        // Align to GGUF_ALIGN before tensor data
        long pos = ftell(fout);
        long align_pad = (GGUF_ALIGN - (pos % GGUF_ALIGN)) % GGUF_ALIGN;
        for (long p = 0; p < align_pad; p++) fputc(0, fout);

        // Write tensor data with alignment
        for (int i = 0; i < n_t; i++) {
            // Align
            pos = ftell(fout);
            long tpad = (GGUF_ALIGN - (pos % GGUF_ALIGN)) % GGUF_ALIGN;
            for (long p = 0; p < tpad; p++) fputc(0, fout);

            fwrite(tensors[i].data, 1, tensors[i].nbytes, fout);
        }

        long bytes = ftell(fout);
        fclose(fout);

        // Free strdup'd names
        for (int i = 1; i < n_t - 1; i++) free((void*)tensors[i].name);

        fprintf(stderr, "GGUF export: %d tensors, %.1f MB\n", n_t, bytes / 1e6);
        char esc_gguf[PATH_MAX*2];
        nf_json_escape(output_path, esc_gguf, sizeof(esc_gguf));
        fprintf(stdout, "{\"type\":\"export\",\"format\":\"gguf\",\"output\":\"%s\","
                "\"bytes\":%ld,\"tensors\":%d,\"step\":%d,\"loss\":%.6f}\n",
                esc_gguf, bytes, n_t, step, loss);
        fflush(stdout);

        #undef GGUF_ALIGN
        #undef GGML_F32
        #undef GGUF_STR
        #undef GGUF_KV_U32
        #undef GGUF_KV_STR
        #undef GGUF_KV_F32
    } else {
        nf_emit_error("unknown format (use llama2c or gguf)", 23);
        return 1;
    }

    // Cleanup
    for (int L=0; L<NLAYERS; L++) { layer_weights_free(&lw[L]); layer_adam_free(&la[L]); }
    free(rms_final); free(embed);
    adam_free(&arms_final); adam_free(&aembed);
    return 0;
}

// ===== Benchmark command =====
static int nf_cmd_benchmark(NFConfig *cfg) {
    int bench_steps = cfg->total_steps > 0 ? cfg->total_steps : 100;

    LayerWeights lw[NLAYERS]; LayerKernels kern[NLAYERS];
    for (int L=0; L<NLAYERS; L++) {
        lw[L] = layer_weights_alloc();
        memset(&kern[L], 0, sizeof(LayerKernels));
    }
    float *rms_final = (float*)malloc(DIM*4);
    float *embed = (float*)malloc(VOCAB*DIM*4);

    if (!load_pretrained(lw, rms_final, embed, cfg->model_path)) {
        nf_emit_error("cannot load model for benchmark", 1);
        return 1;
    }

    // Compile forward-pass kernels
    fprintf(stderr, "Compiling forward kernels for benchmark...\n");
    for (int L=0; L<NLAYERS; L++) {
        if (!compile_layer_kernels(&kern[L], &lw[L])) {
            nf_emit_error("kernel compile failed during benchmark", 4);
            return 1;
        }
    }
    bool ane_extras = cfg->use_ane_extras;
    Kern *finalRmsKern = NULL, *classifierKern = NULL, *softmaxKern = NULL;
    if (ane_extras) {
        finalRmsKern = compile_final_rmsnorm_kern(rms_final);
        classifierKern = compile_classifier_fwd(embed);
        softmaxKern = compile_softmax_kern();
    }

    float *x_cur = (float*)malloc(SEQ*DIM*4);
    float *x_final = (float*)malloc(SEQ*DIM*4);
    float *probs = (float*)malloc(SEQ*VOCAB*4);

    // Generate dummy input
    srand48(cfg->seed);
    uint16_t dummy_tokens[SEQ];
    for (int i=0; i<SEQ; i++) dummy_tokens[i] = (uint16_t)(drand48() * VOCAB);
    embed_lookup(x_cur, embed, dummy_tokens, DIM, SEQ);

    // FLOP constants
    double fwd_flops = NLAYERS * (4.0*2*DIM*DIM*SEQ + 2.0*2*DIM*HIDDEN*SEQ + 2.0*HIDDEN*DIM*SEQ);
    double total_ms = 0;

    fprintf(stderr, "Running %d forward passes...\n", bench_steps);
    for (int s=0; s<bench_steps; s++) {
        uint64_t t0 = mach_absolute_time();

        for (int L=0; L<NLAYERS; L++) {
            LayerKernels *lk = &kern[L];
            io_write_fp16(lk->fwdAttn->ioIn, x_cur, DIM, SEQ);
            ane_eval(lk->fwdAttn);
            float *o_out = x_final; // reuse buffer
            io_read_fp16(lk->fwdAttn->ioOut, o_out, 0, DIM, SEQ);
            float *attn_residual = (float*)malloc(SEQ*DIM*4);
            vDSP_vadd(x_cur, 1, o_out, 1, attn_residual, 1, (vDSP_Length)(SEQ*DIM));

            io_write_fp16(lk->fwdFFN->ioIn, attn_residual, DIM, SEQ);
            ane_eval(lk->fwdFFN);
            float *ffn_out = o_out;
            io_read_fp16(lk->fwdFFN->ioOut, ffn_out, 0, DIM, SEQ);
            vDSP_vadd(attn_residual, 1, ffn_out, 1, x_cur, 1, (vDSP_Length)(SEQ*DIM));
            free(attn_residual);
        }

        if (ane_extras && finalRmsKern && classifierKern && softmaxKern) {
            io_write_fp16(finalRmsKern->ioIn, x_cur, DIM, SEQ);
            ane_eval(finalRmsKern);
            io_read_fp16(finalRmsKern->ioOut, x_final, 0, DIM, SEQ);
            io_write_fp16(classifierKern->ioIn, x_final, DIM, SEQ);
            ane_eval(classifierKern);
            io_copy(softmaxKern->ioIn, 0, classifierKern->ioOut, 0, VOCAB, SEQ);
            ane_eval(softmaxKern);
        }

        double ms = tb_ms(mach_absolute_time() - t0);
        total_ms += ms;
    }

    double avg_ms = total_ms / bench_steps;
    double tflops_ane = (fwd_flops / avg_ms) / 1e9;
    double tflops_total = tflops_ane; // benchmark is forward-only, so ANE ≈ total

    printf("{\"type\":\"benchmark\",\"steps\":%d,\"avg_ms\":%.2f,"
           "\"total_ms\":%.1f,\"tflops_ane\":%.3f,\"tflops_total\":%.3f}\n",
           bench_steps, avg_ms, total_ms, tflops_ane, tflops_total);
    fflush(stdout);

    // Cleanup
    free(x_cur); free(x_final); free(probs);
    for (int L=0; L<NLAYERS; L++) {
        free_layer_kernels(&kern[L]);
        layer_weights_free(&lw[L]);
    }
    if (finalRmsKern) free_kern(finalRmsKern);
    if (classifierKern) free_kern(classifierKern);
    if (softmaxKern) free_kern(softmaxKern);
    free(rms_final); free(embed);
    return 0;
}

// ===== Main entry point =====
static void print_usage(void) {
    fprintf(stderr,
        "NeuralForge — On-device AI training via Apple Neural Engine\n"
        "\n"
        "Usage:\n"
        "  neuralforge train     [options]   Train a model\n"
        "  neuralforge tokenize  [options]   Tokenize text to binary tokens\n"
        "  neuralforge export    [options]   Export checkpoint to model format\n"
        "  neuralforge info      [options]   Show model info\n"
        "  neuralforge benchmark [options]   Benchmark ANE forward pass speed\n"
        "  neuralforge help                  Show this help\n"
        "\n"
        "Train options:\n"
        "  --model PATH      Model weights file (default: stories110M.bin)\n"
        "  --data PATH       Token data file (default: tinystories_data00.bin)\n"
        "  --ckpt PATH       Checkpoint path (default: neuralforge_ckpt.bin)\n"
        "  --steps N         Total training steps (default: 10000)\n"
        "  --lr F            Learning rate (default: 3e-4)\n"
        "  --accum N         Gradient accumulation steps (default: 10)\n"
        "  --ckpt-every N    Checkpoint frequency in steps (default: 100)\n"
        "  --seed N          Random seed (default: 42)\n"
        "  --resume          Resume from checkpoint\n"
        "  --no-ane-extras   Disable ANE for classifier/softmax/rmsnorm_bwd\n"
        "  --no-json         Human-readable output instead of JSON\n"
        "\n"
        "Tokenize options:\n"
        "  --input PATH      Input text file (required)\n"
        "  --output PATH     Output binary token file (default: tokens.bin)\n"
        "  --tokenizer PATH  Tokenizer model file (default: tokenizer.bin)\n"
        "\n"
        "Export options:\n"
        "  --ckpt PATH       Checkpoint to export (required)\n"
        "  --format FMT      Output format: llama2c, gguf (default: llama2c)\n"
        "  --output PATH     Output file path (required)\n"
        "\n"
        "Benchmark options:\n"
        "  --model PATH      Model weights file (default: stories110M.bin)\n"
        "  --steps N         Number of forward passes (default: 100)\n"
        "  --no-ane-extras   Disable ANE extras during benchmark\n"
    );
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        ane_init();
        mach_timebase_info(&g_tb);

        if (argc < 2) { print_usage(); return 1; }

        const char *cmd = argv[1];

        if (strcmp(cmd, "help") == 0 || strcmp(cmd, "--help") == 0 || strcmp(cmd, "-h") == 0) {
            print_usage();
            return 0;
        }

        NFConfig cfg = nf_config_from_args(argc - 2, argv + 2);

        if (strcmp(cmd, "info") == 0) {
            return nf_cmd_info(&cfg);
        }

        if (strcmp(cmd, "train") == 0) {
            return nf_cmd_train(&cfg, argc, argv);
        }

        if (strcmp(cmd, "tokenize") == 0) {
            return nf_cmd_tokenize(argc - 2, argv + 2);
        }

        if (strcmp(cmd, "export") == 0) {
            return nf_cmd_export(argc - 2, argv + 2);
        }

        if (strcmp(cmd, "benchmark") == 0) {
            return nf_cmd_benchmark(&cfg);
        }

        fprintf(stderr, "Unknown command: %s\n", cmd);
        print_usage();
        return 1;
    }
}
