// main.m — NeuralForge CLI entry point
// Commands: train, generate, tokenize, export, info, benchmark, models, download, help
// All progress output is JSON on stdout; logs go to stderr
//
// Build: make neuralforge
// Usage: ./neuralforge train --model stories110M.bin --data tokens.bin --steps 1000
//        ./neuralforge generate --model stories110M.bin --prompt "Once upon a time"
//        ./neuralforge info --model stories110M.bin
//        ./neuralforge train --resume --ckpt checkpoint.bin --data tokens.bin

#include "config.h"
#include "progress.h"
#include "tokenizer.h"
#include "audit.h"
#include "ingest.h"
#include "models.h"

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
// Populates g_mc from the model header (runtime dimensions).
static bool load_pretrained(LayerWeights *lw, float *rms_final, float *embed,
                            const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return false; }
    Llama2Config hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fprintf(stderr, "ERROR: truncated model header in %s\n", path);
        fclose(f); return false;
    }
    if (hdr.vocab_size == INT_MIN) { fclose(f); return false; }  // abs(INT_MIN) is UB
    int V = abs(hdr.vocab_size);
    fprintf(stderr, "  Model: dim=%d hidden=%d layers=%d heads=%d vocab=%d seq=%d\n",
            hdr.dim, hdr.hidden_dim, hdr.n_layers, hdr.n_heads, V, hdr.seq_len);

    // Populate runtime model config from file header
    g_mc.dim = hdr.dim;
    g_mc.hidden_dim = hdr.hidden_dim;
    g_mc.n_heads = hdr.n_heads;
    g_mc.seq_len = hdr.seq_len;
    g_mc.n_layers = hdr.n_layers;
    g_mc.vocab_size = V;
    if (!model_config_validate(&g_mc)) {
        fprintf(stderr, "ERROR: model dimensions out of valid range\n");
        fclose(f); return false;
    }
    model_config_init(&g_mc);

    // Load weights with fread return value checks
    size_t rd = 0, ok = 0;
    rd = fread(embed, 4, (size_t)V * DIM, f); ok += (rd == (size_t)V * DIM);
    for (int L=0; L<NLAYERS; L++) { rd = fread(lw[L].rms_att, 4, DIM, f); ok += (rd == (size_t)DIM); }
    for (int L=0; L<NLAYERS; L++) { rd = fread(lw[L].Wq, 4, WQ_SZ, f); ok += (rd == (size_t)WQ_SZ); }
    for (int L=0; L<NLAYERS; L++) { rd = fread(lw[L].Wk, 4, WQ_SZ, f); ok += (rd == (size_t)WQ_SZ); }
    for (int L=0; L<NLAYERS; L++) { rd = fread(lw[L].Wv, 4, WQ_SZ, f); ok += (rd == (size_t)WQ_SZ); }
    for (int L=0; L<NLAYERS; L++) { rd = fread(lw[L].Wo, 4, WO_SZ, f); ok += (rd == (size_t)WO_SZ); }
    for (int L=0; L<NLAYERS; L++) { rd = fread(lw[L].rms_ffn, 4, DIM, f); ok += (rd == (size_t)DIM); }
    for (int L=0; L<NLAYERS; L++) { rd = fread(lw[L].W1, 4, W1_SZ, f); ok += (rd == (size_t)W1_SZ); }
    for (int L=0; L<NLAYERS; L++) { rd = fread(lw[L].W2, 4, W2_SZ, f); ok += (rd == (size_t)W2_SZ); }
    for (int L=0; L<NLAYERS; L++) { rd = fread(lw[L].W3, 4, W3_SZ, f); ok += (rd == (size_t)W3_SZ); }
    rd = fread(rms_final, 4, DIM, f); ok += (rd == (size_t)DIM);
    size_t expected = 1 + 9 * (size_t)NLAYERS + 1;  // embed + 9 weight groups per layer + rms_final
    if (ok != expected) {
        fprintf(stderr, "ERROR: model file truncated (%zu/%zu reads OK)\n", ok, expected);
        fclose(f); return false;
    }
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
                            float *embed, AdamState *aembed,
                            int lora_rank, float lora_alpha, int lora_targets,
                            LayerLoRA *llora, LayerLoRAAdam *la_lora) {
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
    h.pad[0] = lora_rank;
    h.pad[1] = lora_targets;
    memcpy(&h.pad[2], &lora_alpha, sizeof(float));
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
    // LoRA weights + Adam states (appended after base checkpoint)
    if (lora_rank > 0 && llora && la_lora) {
        for (int L = 0; L < NLAYERS; L++) {
            if (lora_targets & 8) {
                size_t lA = (size_t)lora_rank * DIM;
                size_t lB = (size_t)DIM * lora_rank;
                fwrite(llora[L].wo.A, 4, lA, f);
                fwrite(llora[L].wo.B, 4, lB, f);
                fwrite(la_lora[L].wo.A.m, 4, lA, f);
                fwrite(la_lora[L].wo.A.v, 4, lA, f);
                fwrite(la_lora[L].wo.B.m, 4, lB, f);
                fwrite(la_lora[L].wo.B.v, 4, lB, f);
            }
        }
    }
    fclose(f);
}

static bool load_checkpoint(const char *path, int *step, int *total_steps,
                             float *lr, float *loss,
                             double *cc, double *ct, double *cw,
                             int *cs, int *cb, int *adam_t,
                             LayerWeights *lw, LayerAdam *la,
                             float *rms_final, AdamState *arms_final,
                             float *embed, AdamState *aembed,
                             int lora_rank, int lora_targets,
                             LayerLoRA *llora, LayerLoRAAdam *la_lora) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    CkptHdr h;
    if (fread(&h, sizeof(h), 1, f) != 1) {
        fprintf(stderr, "ERROR: truncated checkpoint header\n");
        fclose(f); return false;
    }
    if (h.magic != 0x424C5A54 || h.version != 2) { fclose(f); return false; }
    // Populate runtime model config from checkpoint header
    g_mc.dim = h.dim;
    g_mc.hidden_dim = h.hidden_dim;
    g_mc.n_heads = h.n_heads;
    g_mc.seq_len = h.seq_len;
    g_mc.n_layers = h.n_layers;
    g_mc.vocab_size = h.vocab_size;
    if (!model_config_validate(&g_mc)) {
        fprintf(stderr, "ERROR: checkpoint dimensions out of valid range\n");
        fclose(f); return false;
    }
    model_config_init(&g_mc);
    *step = h.step; *total_steps = h.total_steps; *lr = h.lr; *loss = h.loss;
    *cc = h.cum_compile; *ct = h.cum_train; *cw = h.cum_wall;
    *cs = h.cum_steps; *cb = h.cum_batches; *adam_t = h.adam_t;

    // Verify checkpoint file size matches expected dimensions before reading
    long cur = ftell(f);
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, cur, SEEK_SET);
    // Sanity check: file must be at least header + some weight data
    size_t expected_min = sizeof(CkptHdr) + (size_t)NLAYERS * (size_t)DIM * 4;
    if (fsize < (long)expected_min) {
        fprintf(stderr, "ERROR: checkpoint file too small (%ld bytes)\n", fsize);
        fclose(f); return false;
    }

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
    // Check for read errors at the end of base data
    if (ferror(f)) {
        fprintf(stderr, "ERROR: checkpoint file read error\n");
        fclose(f); return false;
    }
    // Read LoRA weights + Adam states if checkpoint and config both have LoRA
    int ckpt_lora_rank = h.pad[0];
    int ckpt_lora_targets = h.pad[1];
    if (ckpt_lora_rank > 0 && lora_rank == ckpt_lora_rank && llora && la_lora) {
        if (ckpt_lora_targets != lora_targets) {
            fprintf(stderr, "  Warning: checkpoint LoRA targets=0x%x != config=0x%x\n",
                    ckpt_lora_targets, lora_targets);
        }
        fprintf(stderr, "  Resuming LoRA: rank=%d, targets=0x%x\n", ckpt_lora_rank, ckpt_lora_targets);
        for (int L = 0; L < NLAYERS; L++) {
            if (ckpt_lora_targets & 8) {
                size_t lA = (size_t)lora_rank * DIM;
                size_t lB = (size_t)DIM * lora_rank;
                fread(llora[L].wo.A, 4, lA, f);
                fread(llora[L].wo.B, 4, lB, f);
                fread(la_lora[L].wo.A.m, 4, lA, f);
                fread(la_lora[L].wo.A.v, 4, lA, f);
                fread(la_lora[L].wo.B.m, 4, lB, f);
                fread(la_lora[L].wo.B.v, 4, lB, f);
            }
        }
    } else if (ckpt_lora_rank > 0 && lora_rank != ckpt_lora_rank) {
        fprintf(stderr, "  Warning: checkpoint LoRA rank=%d != config rank=%d, skipping LoRA data\n",
                ckpt_lora_rank, lora_rank);
    }
    fclose(f);
    return true;
}

// ===== Header readers for multi-model support =====
// Read model file header to populate g_mc dimensions (without loading weights)
static bool nf_read_model_header(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    Llama2Config hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return false; }
    fclose(f);
    if (hdr.vocab_size == INT_MIN) return false;
    g_mc.dim = hdr.dim;
    g_mc.hidden_dim = hdr.hidden_dim;
    g_mc.n_heads = hdr.n_heads;
    g_mc.seq_len = hdr.seq_len;
    g_mc.n_layers = hdr.n_layers;
    g_mc.vocab_size = abs(hdr.vocab_size);
    if (!model_config_validate(&g_mc)) return false;
    model_config_init(&g_mc);
    return true;
}

// Read checkpoint header to populate g_mc dimensions (without loading weights)
static bool nf_read_ckpt_header(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    CkptHdr h;
    if (fread(&h, sizeof(h), 1, f) != 1) { fclose(f); return false; }
    fclose(f);
    if (h.magic != 0x424C5A54 || h.version != 2) return false;
    g_mc.dim = h.dim;
    g_mc.hidden_dim = h.hidden_dim;
    g_mc.n_heads = h.n_heads;
    g_mc.seq_len = h.seq_len;
    g_mc.n_layers = h.n_layers;
    g_mc.vocab_size = h.vocab_size;
    if (!model_config_validate(&g_mc)) return false;
    model_config_init(&g_mc);
    return true;
}

// ===== Info command =====
static int nf_cmd_info(NFConfig *cfg) {
    if (!nf_read_model_header(cfg->model_path)) {
        nf_emit_error("cannot open model file", 1);
        return 1;
    }

    fprintf(stdout, "{\"type\":\"info\",\"key\":\"model\",\"value\":{"
            "\"dim\":%d,\"hidden_dim\":%d,\"n_layers\":%d,\"n_heads\":%d,"
            "\"vocab_size\":%d,\"seq_len\":%d,"
            "\"params_millions\":%.1f}}\n",
            DIM, HIDDEN, NLAYERS, HEADS, VOCAB, SEQ,
            (float)TOTAL_PARAMS / 1e6);
    fflush(stdout);
    return 0;
}

// ===== Shuffled data position RNG =====
static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    if (x == 0) x = 1;  // xorshift32 has zero absorbing state — guard against it
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// ===== Learning Rate Scheduler =====
static float nf_compute_lr(NFConfig *cfg, int step) {
    float lr = cfg->learning_rate;
    // Warmup phase: linear ramp from 0 to base LR
    if (cfg->warmup_steps > 0 && step < cfg->warmup_steps) {
        return lr * ((float)(step + 1) / (float)cfg->warmup_steps);
    }
    // Cosine decay phase
    if (cfg->lr_schedule == 1) {
        int decay_steps = cfg->total_steps - cfg->warmup_steps;
        if (decay_steps <= 0) return lr;
        float decay_ratio = (float)(step - cfg->warmup_steps) / (float)decay_steps;
        if (decay_ratio > 1.0f) decay_ratio = 1.0f;
        if (decay_ratio < 0.0f) decay_ratio = 0.0f;
        float coeff = 0.5f * (1.0f + cosf((float)M_PI * decay_ratio));
        return cfg->lr_min + (lr - cfg->lr_min) * coeff;
    }
    // Constant LR (default)
    return lr;
}

// ===== Train command =====
// This is the core training loop extracted from train_large_ane.m
// with JSON progress output and SIGINT handling
static int nf_cmd_train(NFConfig *cfg, int argc, char **argv) {
    signal(SIGINT, sigint_handler);

    // Read model/checkpoint header to populate g_mc before allocating
    if (cfg->resume && cfg->ckpt_path[0] != '\0') {
        nf_read_ckpt_header(cfg->ckpt_path);
    } else if (cfg->model_path[0] != '\0') {
        nf_read_model_header(cfg->model_path);
    }

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
    // LoRA adapters (allocated only if rank > 0)
    int lora_rank = cfg->lora_rank;
    float lora_scale = lora_rank > 0 ? cfg->lora_alpha / (float)lora_rank : 0;
    LayerLoRA llora[NLAYERS];
    LayerLoRAAdam la_lora[NLAYERS];
    LayerLoRAGrad lgrads_lora[NLAYERS];
    if (lora_rank > 0) {
        srand48(cfg->seed + 777);  // Deterministic LoRA init
        for (int L = 0; L < NLAYERS; L++) {
            llora[L] = layer_lora_alloc(lora_rank, cfg->lora_targets);
            la_lora[L] = layer_lora_adam_alloc(lora_rank, cfg->lora_targets);
            lgrads_lora[L] = layer_lora_grad_alloc(lora_rank, cfg->lora_targets);
        }
        srand48(cfg->seed);  // Reset for data shuffling
        int lora_params = 0;
        if (cfg->lora_targets & 8) lora_params += 2 * DIM * lora_rank * NLAYERS;
        fprintf(stderr, "  LoRA: rank=%d, alpha=%.0f, targets=0x%x, params=%d\n",
                lora_rank, cfg->lora_alpha, cfg->lora_targets, lora_params);
    } else {
        memset(llora, 0, sizeof(llora));
        memset(la_lora, 0, sizeof(la_lora));
        memset(lgrads_lora, 0, sizeof(lgrads_lora));
    }

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
                                   lw, la, rms_final, &arms_final, embed, &aembed,
                                   lora_rank, cfg->lora_targets, llora, la_lora);
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

    // mmap validation data (optional)
    uint16_t *val_data = NULL;
    size_t val_n_tokens = 0;
    int val_fd = -1;
    size_t val_data_len = 0;
    bool do_val = (cfg->val_every > 0 && cfg->val_data_path[0] != '\0');
    if (do_val) {
        val_fd = open(cfg->val_data_path, O_RDONLY);
        if (val_fd < 0) {
            fprintf(stderr, "Warning: cannot open val data %s, disabling validation\n",
                    cfg->val_data_path);
            do_val = false;
        } else {
            struct stat vst; fstat(val_fd, &vst);
            val_data_len = vst.st_size;
            val_data = (uint16_t*)mmap(NULL, val_data_len, PROT_READ, MAP_PRIVATE, val_fd, 0);
            if (val_data == MAP_FAILED) {
                fprintf(stderr, "Warning: mmap val data failed, disabling validation\n");
                do_val = false;
                close(val_fd);
                val_fd = -1;
            } else {
                val_n_tokens = val_data_len / 2;
                fprintf(stderr, "  Validation data: %zu tokens\n", val_n_tokens);
            }
        }
    }

    // Shuffle RNG state (xorshift32)
    uint32_t shuffle_rng = (uint32_t)(cfg->seed + 12345);

    nf_emit_init(TOTAL_PARAMS, NLAYERS, DIM, HIDDEN, HEADS, SEQ, VOCAB);
    if (lora_rank > 0) {
        int lp = 0;
        if (cfg->lora_targets & 8) lp += 2 * DIM * lora_rank * NLAYERS;
        nf_emit_lora_info(lora_rank, cfg->lora_alpha, cfg->lora_targets, lp);
    }

    // Audit: training start
    nf_audit_training_start(cfg->model_path, cfg->data_path,
                            cfg->total_steps, cfg->learning_rate,
                            cfg->accum_steps, lora_rank, cfg->lora_alpha);

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
                lw, la, rms_final, &arms_final, embed, &aembed,
                lora_rank, cfg->lora_alpha, cfg->lora_targets, llora, la_lora);

            nf_emit_restart(step, g_compile_count, "compile_budget_exceeded");

            // exec() restart — PID stays same, pipes stay open
            char steps_str[32]; snprintf(steps_str, sizeof(steps_str), "%d", total_steps);
            char lr_str[32]; snprintf(lr_str, sizeof(lr_str), "%.6e", cfg->learning_rate);
            char warmup_str[32]; snprintf(warmup_str, sizeof(warmup_str), "%d", cfg->warmup_steps);
            char lr_min_str[32]; snprintf(lr_min_str, sizeof(lr_min_str), "%.6e", cfg->lr_min);
            const char *sched_str = cfg->lr_schedule == 1 ? "cosine" : "none";
            char val_every_str[32]; snprintf(val_every_str, sizeof(val_every_str), "%d", cfg->val_every);
            char val_batches_str[32]; snprintf(val_batches_str, sizeof(val_batches_str), "%d", cfg->val_batches);

            // Build exec args dynamically to handle optional flags
            const char *exec_args[50];
            int ea = 0;
            exec_args[ea++] = argv[0];
            exec_args[ea++] = "train";
            exec_args[ea++] = "--resume";
            if (!ane_extras) exec_args[ea++] = "--no-ane-extras";
            exec_args[ea++] = "--ckpt"; exec_args[ea++] = cfg->ckpt_path;
            exec_args[ea++] = "--data"; exec_args[ea++] = cfg->data_path;
            exec_args[ea++] = "--steps"; exec_args[ea++] = steps_str;
            exec_args[ea++] = "--lr"; exec_args[ea++] = lr_str;
            exec_args[ea++] = "--warmup"; exec_args[ea++] = warmup_str;
            exec_args[ea++] = "--lr-min"; exec_args[ea++] = lr_min_str;
            exec_args[ea++] = "--lr-schedule"; exec_args[ea++] = sched_str;
            if (cfg->shuffle) exec_args[ea++] = "--shuffle";
            if (do_val) {
                exec_args[ea++] = "--val-data"; exec_args[ea++] = cfg->val_data_path;
                exec_args[ea++] = "--val-every"; exec_args[ea++] = val_every_str;
                exec_args[ea++] = "--val-batches"; exec_args[ea++] = val_batches_str;
            }
            char lora_rank_str[32], lora_alpha_str[32], lora_targets_str[32];
            if (lora_rank > 0) {
                snprintf(lora_rank_str, sizeof(lora_rank_str), "%d", lora_rank);
                snprintf(lora_alpha_str, sizeof(lora_alpha_str), "%.1f", cfg->lora_alpha);
                snprintf(lora_targets_str, sizeof(lora_targets_str), "%d", cfg->lora_targets);
                exec_args[ea++] = "--lora-rank"; exec_args[ea++] = lora_rank_str;
                exec_args[ea++] = "--lora-alpha"; exec_args[ea++] = lora_alpha_str;
                exec_args[ea++] = "--lora-targets"; exec_args[ea++] = lora_targets_str;
            }
            exec_args[ea] = NULL;
            execv(argv[0], (char *const *)exec_args);
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
        for (int L=0; L<NLAYERS; L++) {
            layer_grads_zero(&grads[L]);
            if (lora_rank > 0) layer_lora_grad_zero(&lgrads_lora[L], lora_rank, cfg->lora_targets);
        }
        memset(grms_final, 0, DIM*4);
        memset(gembed, 0, (size_t)VOCAB*DIM*4);

        int steps_batch = 0;
        uint64_t tt = mach_absolute_time();

        for (int a=0; a<cfg->accum_steps && step<total_steps && !g_interrupted; a++, step++) {
            if (n_tokens <= (size_t)SEQ + 1) {
                nf_emit_error("token data too small for sequence length", 2);
                return 1;
            }
            size_t max_pos = n_tokens - SEQ - 1;
            size_t pos;
            if (cfg->shuffle) {
                pos = (size_t)(xorshift32(&shuffle_rng) % max_pos);
            } else {
                pos = (size_t)(drand48() * max_pos);
            }
            uint16_t *input_tokens = token_data + pos;
            uint16_t *target_tokens = token_data + pos + 1;

            // Embedding lookup
            embed_lookup(x_cur, embed, input_tokens, DIM, SEQ, VOCAB);

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

                // LoRA: add Wo adapter delta to o_out
                if (lora_rank > 0 && (cfg->lora_targets & 8)) {
                    lora_forward(ac->o_out, ac->attn_out,
                                 llora[L].wo.A, llora[L].wo.B,
                                 DIM, DIM, lora_rank, lora_scale, SEQ);
                }

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
                if (tgt < 0 || tgt >= VOCAB) continue;  // bounds check — prevents OOB from bad token data
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

                if (lora_rank == 0) {
                    // Full fine-tune: compute FFN weight gradients
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
                }

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

                // LoRA backward for Wo: compute grad_A, grad_B
                if (lora_rank > 0 && (cfg->lora_targets & 8)) {
                    lora_backward(lgrads_lora[L].wo.A, lgrads_lora[L].wo.B,
                                  do_out_buf, ac->attn_out,
                                  llora[L].wo.A, llora[L].wo.B,
                                  DIM, DIM, lora_rank, lora_scale, SEQ);
                }

                if (lora_rank == 0) {
                    // Full fine-tune: compute base weight gradient for Wo
                    float *capt_do = (float*)malloc(SEQ*DIM*4); memcpy(capt_do, do_out_buf, SEQ*DIM*4);
                    float *capt_attn = (float*)malloc(SEQ*DIM*4); memcpy(capt_attn, ac->attn_out, SEQ*DIM*4);
                    dispatch_group_async(dw_grp, dw_q, ^{
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                    1.0f, capt_do, SEQ, capt_attn, SEQ, 1.0f, gr->Wo, DIM);
                        free(capt_do); free(capt_attn);
                    });
                }

                io_copy(kern[L].sdpaBwd1->ioIn, 0, kern[L].fwdAttn->ioOut, DIM, 3*DIM, SEQ);
                io_write_fp16_at(kern[L].sdpaBwd1->ioIn, 3*DIM, dx2, DIM, SEQ);
                ane_eval(kern[L].sdpaBwd1);
                io_copy(sdpaBwd2[L]->ioIn, 0, kern[L].sdpaBwd1->ioOut, DIM, 2*SCORE_CH, SEQ);
                io_copy(sdpaBwd2[L]->ioIn, 2*SCORE_CH, kern[L].fwdAttn->ioOut, DIM, 2*DIM, SEQ);
                ane_eval(sdpaBwd2[L]);

                io_read_fp16(sdpaBwd2[L]->ioOut, dq, 0, DIM, SEQ);
                io_read_fp16(sdpaBwd2[L]->ioOut, dk, DIM, DIM, SEQ);
                io_read_fp16(kern[L].sdpaBwd1->ioOut, dv, 0, DIM, SEQ);

                if (lora_rank == 0) {
                    // Full fine-tune: compute Q/K/V weight gradients
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
                }

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
            if (lora_rank == 0) {
                embed_backward(gembed, dy, input_tokens, DIM, SEQ, VOCAB);
            }
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

        // Compute scheduled LR for this step
        lr = nf_compute_lr(cfg, step);

        // Adam update
        float gsc = 1.0f / steps_batch;
        adam_t++;

        if (lora_rank > 0) {
            // LoRA mode: update only LoRA adapters + norm weights (base weights frozen)
            for (int L=0; L<NLAYERS; L++) {
                LayerGrads *g = &grads[L];
                // Scale norm gradients
                for(int i=0;i<DIM;i++){g->rms_att[i]*=gsc; g->rms_ffn[i]*=gsc;}
                adam_update(lw[L].rms_att, g->rms_att, &la[L].rms_att, adam_t, lr, cfg->beta1, cfg->beta2, cfg->eps);
                adam_update(lw[L].rms_ffn, g->rms_ffn, &la[L].rms_ffn, adam_t, lr, cfg->beta1, cfg->beta2, cfg->eps);
                // LoRA Wo adapter
                if (cfg->lora_targets & 8) {
                    size_t lA_sz = (size_t)lora_rank * DIM;
                    size_t lB_sz = (size_t)DIM * lora_rank;
                    for(size_t i=0;i<lA_sz;i++) lgrads_lora[L].wo.A[i] *= gsc;
                    for(size_t i=0;i<lB_sz;i++) lgrads_lora[L].wo.B[i] *= gsc;
                    adam_update(llora[L].wo.A, lgrads_lora[L].wo.A, &la_lora[L].wo.A, adam_t, lr, cfg->beta1, cfg->beta2, cfg->eps);
                    adam_update(llora[L].wo.B, lgrads_lora[L].wo.B, &la_lora[L].wo.B, adam_t, lr, cfg->beta1, cfg->beta2, cfg->eps);
                }
            }
            for(int i=0;i<DIM;i++) grms_final[i]*=gsc;
            adam_update(rms_final, grms_final, &arms_final, adam_t, lr, cfg->beta1, cfg->beta2, cfg->eps);
            // Embedding frozen in LoRA mode
        } else {
            // Full fine-tune: update all weights
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
        }

        nf_emit_batch(total_batches, step, last_loss, best_loss, cms, tms, g_compile_count);

        // Periodic checkpoint
        if (step > 0 && step % cfg->checkpoint_every == 0) {
            double wall = tb_ms(mach_absolute_time() - t_wall_start);
            save_checkpoint(cfg->ckpt_path, step, total_steps, lr, last_loss,
                total_compile_ms+cum_compile, total_train_ms+cum_train, wall+cum_wall,
                total_steps_done+cum_steps, total_batches+cum_batches, adam_t,
                lw, la, rms_final, &arms_final, embed, &aembed,
                lora_rank, cfg->lora_alpha, cfg->lora_targets, llora, la_lora);
            nf_emit_checkpoint(cfg->ckpt_path, step, last_loss);
            nf_audit_checkpoint(cfg->ckpt_path, step, last_loss);
        }

        // Periodic validation (forward-only, no backward pass)
        if (do_val && step > 0 && total_batches % cfg->val_every == 0) {
            float val_total_loss = 0;
            uint32_t val_rng = (uint32_t)(cfg->seed + step + 99999);
            if (val_rng == 0) val_rng = 1;
            int actual_val_batches = cfg->val_batches;
            if (val_n_tokens <= (size_t)SEQ + 1) {
                fprintf(stderr, "  Validation data too small, skipping\n");
                continue;
            }
            for (int vb = 0; vb < actual_val_batches; vb++) {
                size_t vmax = val_n_tokens - SEQ - 1;
                size_t vpos = (size_t)(xorshift32(&val_rng) % vmax);
                uint16_t *vinput = val_data + vpos;
                uint16_t *vtarget = val_data + vpos + 1;

                // Embedding
                embed_lookup(x_cur, embed, vinput, DIM, SEQ, VOCAB);

                // Forward pass through all layers
                for (int L = 0; L < NLAYERS; L++) {
                    memcpy(acts[L].layer_in, x_cur, SEQ * DIM * 4);
                    dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
                    io_write_fp16(kern[L].fwdAttn->ioIn, x_cur, DIM, SEQ);
                    ane_eval(kern[L].fwdAttn);
                    io_read_fp16(kern[L].fwdAttn->ioOut, acts[L].o_out, 0, DIM, SEQ);
                    // LoRA: add Wo adapter delta during validation
                    if (lora_rank > 0 && (cfg->lora_targets & 8)) {
                        io_read_fp16(kern[L].fwdAttn->ioOut, acts[L].attn_out, 4*DIM, DIM, SEQ);
                        lora_forward(acts[L].o_out, acts[L].attn_out,
                                     llora[L].wo.A, llora[L].wo.B,
                                     DIM, DIM, lora_rank, lora_scale, SEQ);
                    }
                    vDSP_vadd(x_cur, 1, acts[L].o_out, 1, acts[L].x2, 1, (vDSP_Length)(SEQ * DIM));
                    io_write_fp16(kern[L].fwdFFN->ioIn, acts[L].x2, DIM, SEQ);
                    ane_eval(kern[L].fwdFFN);
                    io_read_fp16(kern[L].fwdFFN->ioOut, acts[L].ffn_out, 0, DIM, SEQ);
                    vDSP_vadd(acts[L].x2, 1, acts[L].ffn_out, 1, x_cur, 1, (vDSP_Length)(SEQ * DIM));
                }

                // Final rmsnorm + classifier (CPU path for simplicity)
                rmsnorm(x_final, x_cur, rms_final, DIM, SEQ);
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            VOCAB, SEQ, DIM, 1.0f, embed, DIM, x_final, SEQ, 0.0f, probs, SEQ);
                // Softmax + NLL loss
                float vloss = 0;
                for (int t = 0; t < SEQ; t++) {
                    float maxv = -1e30f;
                    for (int v = 0; v < VOCAB; v++) { float val = probs[v * SEQ + t]; if (val > maxv) maxv = val; }
                    float sum = 0;
                    for (int v = 0; v < VOCAB; v++) { probs[v * SEQ + t] = expf(probs[v * SEQ + t] - maxv); sum += probs[v * SEQ + t]; }
                    for (int v = 0; v < VOCAB; v++) probs[v * SEQ + t] /= sum;
                    int vtgt = vtarget[t];
                    if (vtgt >= 0 && vtgt < VOCAB)
                        vloss -= logf(probs[vtgt * SEQ + t] + 1e-10f);
                }
                val_total_loss += vloss / SEQ;
            }
            float avg_val_loss = val_total_loss / actual_val_batches;
            nf_emit_val(step, avg_val_loss, actual_val_batches);
        }
    }

    // Handle SIGINT graceful shutdown
    if (g_interrupted) {
        double wall = tb_ms(mach_absolute_time() - t_wall_start);
        save_checkpoint(cfg->ckpt_path, step, total_steps, lr, last_loss,
            total_compile_ms+cum_compile, total_train_ms+cum_train, wall+cum_wall,
            total_steps_done+cum_steps, total_batches+cum_batches, adam_t,
            lw, la, rms_final, &arms_final, embed, &aembed,
            lora_rank, cfg->lora_alpha, cfg->lora_targets, llora, la_lora);
        nf_emit_checkpoint(cfg->ckpt_path, step, last_loss);
        nf_audit_checkpoint(cfg->ckpt_path, step, last_loss);
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
    nf_audit_training_stop(total_steps_done, last_loss, wall / 1000.0,
                           g_interrupted ? "interrupted" : "completed");
    nf_audit_close();

    // Cleanup
    for (int L=0; L<NLAYERS; L++) {
        free_layer_kernels(&kern[L]); free_kern(sdpaBwd2[L]);
        free_kern(rmsAttBwd[L]); free_kern(rmsFFNBwd[L]);
        layer_weights_free(&lw[L]); layer_adam_free(&la[L]);
        layer_acts_free(&acts[L]); layer_grads_free(&grads[L]);
        if (lora_rank > 0) {
            layer_lora_free(&llora[L], cfg->lora_targets);
            layer_lora_adam_free(&la_lora[L], cfg->lora_targets);
            layer_lora_grad_free(&lgrads_lora[L], cfg->lora_targets);
        }
    }
    free_kern(softmaxKern); free_kern(finalRmsKern); free_kern(classifierKern);
    munmap(token_data, data_len); close(data_fd);
    if (val_data) { munmap(val_data, val_data_len); close(val_fd); }
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

    // Read checkpoint header to populate g_mc before allocating
    nf_read_ckpt_header(ckpt_path);

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
                          lw, la, rms_final, &arms_final, embed, &aembed,
                          0, 0, NULL, NULL)) {
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

    // Audit: export event
    nf_audit_export(ckpt_path, output_path, format);

    // Cleanup
    for (int L=0; L<NLAYERS; L++) { layer_weights_free(&lw[L]); layer_adam_free(&la[L]); }
    free(rms_final); free(embed);
    adam_free(&arms_final); adam_free(&aembed);
    return 0;
}

// ===== Benchmark command =====
static int nf_cmd_benchmark(NFConfig *cfg) {
    int bench_steps = cfg->total_steps > 0 ? cfg->total_steps : 100;

    // Read model header to populate g_mc before allocating
    nf_read_model_header(cfg->model_path);

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
    embed_lookup(x_cur, embed, dummy_tokens, DIM, SEQ, VOCAB);

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

// ===== Generate command (text generation / inference) =====

typedef struct { float prob; int idx; } ProbIndex;

static int prob_compare_desc(const void *a, const void *b) {
    float pa = ((const ProbIndex *)a)->prob;
    float pb = ((const ProbIndex *)b)->prob;
    if (pa > pb) return -1;
    if (pa < pb) return 1;
    return 0;
}

static int sample_topp(float *probs, int n, float topp, uint32_t *rng) {
    ProbIndex *pi = (ProbIndex *)malloc(n * sizeof(ProbIndex));
    for (int i = 0; i < n; i++) { pi[i].prob = probs[i]; pi[i].idx = i; }
    qsort(pi, n, sizeof(ProbIndex), prob_compare_desc);

    // Accumulate until top_p
    float cum = 0;
    int cutoff = 0;
    for (int i = 0; i < n; i++) {
        cum += pi[i].prob;
        cutoff = i;
        if (cum >= topp) break;
    }

    // Renormalize and sample
    float sub_sum = 0;
    for (int i = 0; i <= cutoff; i++) sub_sum += pi[i].prob;
    float r = (float)xorshift32(rng) / 4294967296.0f * sub_sum;
    float cdf = 0;
    int token = pi[0].idx;
    for (int i = 0; i <= cutoff; i++) {
        cdf += pi[i].prob;
        if (cdf >= r) { token = pi[i].idx; break; }
    }

    free(pi);
    return token;
}

static int nf_cmd_generate(int argc, char **argv) {
    const char *model_path = "stories110M.bin";
    const char *tokenizer_path = "tokenizer.bin";
    const char *prompt = "Once upon a time";
    float temperature = 0.8f;
    float top_p = 0.9f;
    int max_tokens = 256;
    int seed = 42;

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i+1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--tokenizer") == 0 && i+1 < argc) tokenizer_path = argv[++i];
        else if (strcmp(argv[i], "--prompt") == 0 && i+1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "--temperature") == 0 && i+1 < argc)
            temperature = nf_safe_atof(argv[++i], 0.8f, 0.01f, 5.0f);
        else if (strcmp(argv[i], "--top-p") == 0 && i+1 < argc)
            top_p = nf_safe_atof(argv[++i], 0.9f, 0.0f, 1.0f);
        else if (strcmp(argv[i], "--max-tokens") == 0 && i+1 < argc)
            max_tokens = nf_safe_atoi(argv[++i], 256, 1, SEQ);
        else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc)
            seed = nf_safe_atoi(argv[++i], 42, 0, INT_MAX - 1);
    }

    // Read model header to populate g_mc before allocating
    nf_read_model_header(model_path);

    // Re-validate max_tokens with actual model SEQ now that header is loaded
    if (max_tokens > SEQ) max_tokens = SEQ;

    // Load model
    LayerWeights lw[NLAYERS];
    for (int L = 0; L < NLAYERS; L++) lw[L] = layer_weights_alloc();
    float *rms_final = (float *)malloc(DIM * 4);
    float *embed = (float *)malloc(VOCAB * DIM * 4);

    if (!load_pretrained(lw, rms_final, embed, model_path)) {
        nf_emit_error("cannot load model", 1);
        return 1;
    }

    // Load tokenizer
    NFTokenizer tok;
    if (!nf_tokenizer_load(&tok, tokenizer_path, VOCAB)) {
        nf_emit_error("cannot load tokenizer", 11);
        return 1;
    }

    // Tokenize prompt
    uint16_t tokens[SEQ];
    memset(tokens, 0, sizeof(tokens));
    int n_prompt = nf_tokenizer_encode(&tok, prompt, tokens, SEQ - 1);
    if (n_prompt == 0) {
        nf_emit_error("prompt tokenization failed", 30);
        return 1;
    }
    fprintf(stderr, "Prompt: %d tokens\n", n_prompt);

    // Audit: generate start
    nf_audit_generate_start(model_path, max_tokens, temperature, top_p);

    // Compile forward-pass kernels
    fprintf(stderr, "Compiling forward kernels...\n");
    LayerKernels kern[NLAYERS];
    memset(kern, 0, sizeof(kern));
    for (int L = 0; L < NLAYERS; L++) {
        if (!compile_layer_kernels(&kern[L], &lw[L])) {
            nf_emit_error("kernel compile failed", 4);
            return 1;
        }
    }
    fprintf(stderr, "Forward kernels compiled (%d compiles)\n", g_compile_count);

    // Allocate buffers
    float *x_cur = (float *)malloc(SEQ * DIM * 4);
    float *x_final = (float *)malloc(SEQ * DIM * 4);
    float *x2 = (float *)malloc(SEQ * DIM * 4);
    float *logits_buf = (float *)malloc(VOCAB * 4);
    float *h_pos = (float *)malloc(DIM * 4);

    uint32_t rng = (uint32_t)(seed + 7777);
    uint64_t t_start = mach_absolute_time();
    int pos = n_prompt;
    int tokens_generated = 0;

    // Emit prompt tokens
    for (int i = 0; i < n_prompt; i++) {
        const char *text = nf_tokenizer_decode(&tok, tokens[i]);
        nf_emit_token(tokens[i], text ? text : "");
    }

    // Autoregressive generation loop
    while (pos < SEQ && tokens_generated < max_tokens) {
        // Embedding lookup
        embed_lookup(x_cur, embed, tokens, DIM, SEQ, VOCAB);

        // Forward through all layers (ANE)
        for (int L = 0; L < NLAYERS; L++) {
            float *o_out = x_final;  // reuse buffer temporarily
            io_write_fp16(kern[L].fwdAttn->ioIn, x_cur, DIM, SEQ);
            ane_eval(kern[L].fwdAttn);
            io_read_fp16(kern[L].fwdAttn->ioOut, o_out, 0, DIM, SEQ);

            vDSP_vadd(x_cur, 1, o_out, 1, x2, 1, (vDSP_Length)(SEQ * DIM));

            io_write_fp16(kern[L].fwdFFN->ioIn, x2, DIM, SEQ);
            ane_eval(kern[L].fwdFFN);
            float *ffn_out = o_out;
            io_read_fp16(kern[L].fwdFFN->ioOut, ffn_out, 0, DIM, SEQ);

            vDSP_vadd(x2, 1, ffn_out, 1, x_cur, 1, (vDSP_Length)(SEQ * DIM));
        }

        // Final rmsnorm (CPU)
        rmsnorm(x_final, x_cur, rms_final, DIM, SEQ);

        // Extract hidden state at generation position (pos-1) and compute logits
        int gen_pos = pos - 1;
        for (int d = 0; d < DIM; d++) h_pos[d] = x_final[d * SEQ + gen_pos];
        cblas_sgemv(CblasRowMajor, CblasNoTrans, VOCAB, DIM,
                    1.0f, embed, DIM, h_pos, 1, 0.0f, logits_buf, 1);

        // Apply temperature
        if (temperature != 1.0f) {
            float inv_temp = 1.0f / temperature;
            for (int v = 0; v < VOCAB; v++) logits_buf[v] *= inv_temp;
        }

        // Softmax
        float max_logit = -1e30f;
        for (int v = 0; v < VOCAB; v++)
            if (logits_buf[v] > max_logit) max_logit = logits_buf[v];
        float sum = 0;
        for (int v = 0; v < VOCAB; v++) {
            logits_buf[v] = expf(logits_buf[v] - max_logit);
            sum += logits_buf[v];
        }
        for (int v = 0; v < VOCAB; v++) logits_buf[v] /= sum;

        // Sample next token
        int next_token;
        if (top_p >= 1.0f) {
            // Sample from full distribution
            float r = (float)xorshift32(&rng) / 4294967296.0f;
            float cdf = 0;
            next_token = 0;
            for (int v = 0; v < VOCAB; v++) {
                cdf += logits_buf[v];
                if (cdf >= r) { next_token = v; break; }
            }
        } else {
            next_token = sample_topp(logits_buf, VOCAB, top_p, &rng);
        }

        // EOS check (token 2 = </s> in llama tokenizer)
        if (next_token == 2) break;

        // Append token to sequence
        tokens[pos] = (uint16_t)next_token;
        pos++;
        tokens_generated++;

        // Emit token
        const char *text = nf_tokenizer_decode(&tok, next_token);
        nf_emit_token(next_token, text ? text : "");
    }

    double total_ms = tb_ms(mach_absolute_time() - t_start);
    nf_emit_generate_done(tokens_generated, total_ms);
    nf_audit_generate_done(tokens_generated, total_ms);
    fprintf(stderr, "Generated %d tokens in %.1f ms (%.1f ms/token)\n",
            tokens_generated, total_ms,
            tokens_generated > 0 ? total_ms / tokens_generated : 0);

    // Cleanup
    for (int L = 0; L < NLAYERS; L++) {
        free_layer_kernels(&kern[L]);
        layer_weights_free(&lw[L]);
    }
    free(rms_final); free(embed);
    free(x_cur); free(x_final); free(x2);
    free(logits_buf); free(h_pos);
    nf_tokenizer_free(&tok);

    return 0;
}

// ===== Ingest command =====
static int nf_cmd_ingest(int argc, char **argv) {
    const char *source_dir = NULL;
    const char *output_dir = NULL;
    const char *tokenizer_path = "tokenizer.bin";
    const char *manifest_path = NULL;  // defaults to output_dir/manifest.json
    int max_shard_mb = 50;
    bool incremental = false;

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--source") == 0 && i+1 < argc) source_dir = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i+1 < argc) output_dir = argv[++i];
        else if (strcmp(argv[i], "--tokenizer") == 0 && i+1 < argc) tokenizer_path = argv[++i];
        else if (strcmp(argv[i], "--manifest") == 0 && i+1 < argc) manifest_path = argv[++i];
        else if (strcmp(argv[i], "--max-shard-mb") == 0 && i+1 < argc)
            max_shard_mb = nf_safe_atoi(argv[++i], 50, 1, 10000);
        else if (strcmp(argv[i], "--incremental") == 0) incremental = true;
    }

    if (!source_dir) {
        nf_emit_error("--source directory required", 20);
        return 1;
    }
    if (!output_dir) {
        nf_emit_error("--output directory required", 21);
        return 1;
    }

    // Default manifest path: output_dir/manifest.json
    char manifest_buf[PATH_MAX];
    if (!manifest_path) {
        snprintf(manifest_buf, sizeof(manifest_buf), "%s/manifest.json", output_dir);
        manifest_path = manifest_buf;
    }

    // Create output directory if needed
    @autoreleasepool {
        NSFileManager *fm = [NSFileManager defaultManager];
        NSError *err = nil;
        [fm createDirectoryAtPath:[NSString stringWithUTF8String:output_dir]
      withIntermediateDirectories:YES attributes:nil error:&err];
        if (err) {
            nf_emit_error("cannot create output directory", 22);
            return 1;
        }
    }

    // Load tokenizer
    NFTokenizer tok;
    if (!nf_tokenizer_load(&tok, tokenizer_path, VOCAB)) {
        nf_emit_error("cannot load tokenizer", 11);
        return 1;
    }
    fprintf(stderr, "Loaded tokenizer: %d vocab, max_len=%d\n",
            tok.vocab_size, tok.max_token_length);

    // Load existing manifest (for incremental mode)
    NSDictionary *existing_manifest = NULL;
    if (incremental) {
        existing_manifest = nf_manifest_load(manifest_path);
        if (existing_manifest) {
            int prev_tokens = [existing_manifest[@"total_tokens"] intValue];
            fprintf(stderr, "Loaded manifest: %d total tokens\n", prev_tokens);
        }
    }

    // Scan source directory
    NSArray<NSString *> *all_files = nf_scan_source_dir(source_dir);
    fprintf(stderr, "Found %lu supported files in %s\n",
            (unsigned long)all_files.count, source_dir);

    if (all_files.count == 0) {
        nf_emit_ingest_done(0, 0, 0, 0, manifest_path);
        nf_tokenizer_free(&tok);
        return 0;
    }

    // Filter files for incremental mode
    NSMutableArray<NSString *> *files_to_process = [NSMutableArray array];
    int skipped = 0;
    for (NSString *path in all_files) {
        if (incremental && !nf_should_process([path UTF8String], existing_manifest)) {
            skipped++;
        } else {
            [files_to_process addObject:path];
        }
    }

    fprintf(stderr, "Processing %lu files (%d skipped)\n",
            (unsigned long)files_to_process.count, skipped);

    if (files_to_process.count == 0) {
        // Nothing new to process — carry forward existing manifest stats
        int existing_tokens = [existing_manifest[@"total_tokens"] intValue];
        int existing_shards = (int)((NSArray *)existing_manifest[@"shards"]).count;
        nf_emit_ingest_done(0, skipped, existing_tokens, existing_shards, manifest_path);
        nf_tokenizer_free(&tok);
        return 0;
    }

    // Shard size limit in tokens
    int max_shard_tokens = max_shard_mb * 1024 * 1024 / (int)sizeof(uint16_t);

    // Accumulate existing shard info
    NSMutableArray *shard_info = [NSMutableArray array];
    NSMutableDictionary *processed_files = [NSMutableDictionary dictionary];
    int total_tokens = 0;
    int shard_index = 0;

    // Carry forward existing manifest data
    if (existing_manifest) {
        NSArray *existing_shards = existing_manifest[@"shards"];
        if (existing_shards) {
            [shard_info addObjectsFromArray:existing_shards];
            shard_index = (int)existing_shards.count;
        }
        NSDictionary *existing_processed = existing_manifest[@"processed_files"];
        if (existing_processed) {
            [processed_files addEntriesFromDictionary:existing_processed];
        }
        total_tokens = [existing_manifest[@"total_tokens"] intValue];
    }

    // Token accumulation buffer for current shard
    int shard_buf_cap = (max_shard_tokens < 1000000) ? max_shard_tokens : 1000000;
    uint16_t *shard_buf = (uint16_t *)malloc(shard_buf_cap * sizeof(uint16_t));
    if (!shard_buf) {
        nf_emit_error("out of memory for shard buffer", 23);
        nf_tokenizer_free(&tok);
        return 1;
    }
    int shard_buf_len = 0;
    NSMutableArray *current_shard_sources = [NSMutableArray array];
    int new_files = 0;

    // Process each file
    for (NSString *file_path in files_to_process) {
        const char *cpath = [file_path UTF8String];
        const char *filename = [[file_path lastPathComponent] UTF8String];

        fprintf(stderr, "  Processing: %s\n", filename);

        // Extract text
        char *text = nf_extract_text(cpath);
        if (!text) {
            fprintf(stderr, "  Warning: cannot extract text from %s (skipping)\n", filename);
            continue;
        }

        size_t text_len = strlen(text);
        if (text_len == 0) {
            free(text);
            continue;
        }

        // Tokenize in chunks to handle large files
        size_t max_toks = text_len;  // Upper bound: 1 token per byte
        uint16_t *file_tokens = (uint16_t *)malloc(max_toks * sizeof(uint16_t));
        if (!file_tokens) {
            fprintf(stderr, "  Warning: out of memory for %s (skipping)\n", filename);
            free(text);
            continue;
        }

        int file_tok_count = nf_tokenizer_encode(&tok, text, file_tokens, (int)max_toks);
        free(text);

        if (file_tok_count <= 0) {
            free(file_tokens);
            continue;
        }

        nf_emit_ingest_file(filename, file_tok_count);
        fprintf(stderr, "  → %d tokens (%.1f bytes/token)\n",
                file_tok_count, (float)text_len / file_tok_count);

        // Record processed file
        const char *mtime = nf_file_mtime_iso(cpath);
        @autoreleasepool {
            processed_files[[NSString stringWithUTF8String:filename]] = @{
                @"mtime": mtime ? [NSString stringWithUTF8String:mtime] : @"unknown",
                @"tokens": @(file_tok_count)
            };
        }

        // Append tokens to shard buffer, flushing when full
        int offset = 0;
        while (offset < file_tok_count) {
            int space = max_shard_tokens - shard_buf_len;
            int chunk = file_tok_count - offset;
            if (chunk > space) chunk = space;

            // Grow buffer if needed
            if (shard_buf_len + chunk > shard_buf_cap) {
                shard_buf_cap = shard_buf_len + chunk + 100000;
                uint16_t *new_buf = (uint16_t *)realloc(shard_buf,
                                                         shard_buf_cap * sizeof(uint16_t));
                if (!new_buf) {
                    fprintf(stderr, "  Warning: realloc failed, flushing early\n");
                    break;
                }
                shard_buf = new_buf;
            }

            memcpy(shard_buf + shard_buf_len, file_tokens + offset, chunk * sizeof(uint16_t));
            shard_buf_len += chunk;
            offset += chunk;

            [current_shard_sources addObject:[NSString stringWithUTF8String:filename]];

            // Flush shard if full
            if (shard_buf_len >= max_shard_tokens) {
                char shard_path[PATH_MAX], shard_name_buf[64];
                nf_shard_name(shard_index, shard_name_buf, sizeof(shard_name_buf));
                snprintf(shard_path, sizeof(shard_path), "%s/%s", output_dir, shard_name_buf);

                size_t written = nf_write_shard(shard_buf, shard_buf_len, shard_path);
                fprintf(stderr, "  Wrote shard %s: %d tokens (%zu bytes)\n",
                        shard_name_buf, shard_buf_len, written);

                @autoreleasepool {
                    // Deduplicate source files
                    NSOrderedSet *unique = [NSOrderedSet orderedSetWithArray:current_shard_sources];
                    [shard_info addObject:@{
                        @"path": [NSString stringWithUTF8String:shard_name_buf],
                        @"tokens": @(shard_buf_len),
                        @"bytes": @(written),
                        @"source_files": [unique array]
                    }];
                }

                total_tokens += shard_buf_len;
                shard_buf_len = 0;
                shard_index++;
                [current_shard_sources removeAllObjects];
            }
        }

        free(file_tokens);
        new_files++;
    }

    // Flush remaining tokens as final shard
    if (shard_buf_len > 0) {
        char shard_path[PATH_MAX], shard_name_buf[64];
        nf_shard_name(shard_index, shard_name_buf, sizeof(shard_name_buf));
        snprintf(shard_path, sizeof(shard_path), "%s/%s", output_dir, shard_name_buf);

        size_t written = nf_write_shard(shard_buf, shard_buf_len, shard_path);
        fprintf(stderr, "  Wrote shard %s: %d tokens (%zu bytes)\n",
                shard_name_buf, shard_buf_len, written);

        @autoreleasepool {
            NSOrderedSet *unique = [NSOrderedSet orderedSetWithArray:current_shard_sources];
            [shard_info addObject:@{
                @"path": [NSString stringWithUTF8String:shard_name_buf],
                @"tokens": @(shard_buf_len),
                @"bytes": @(written),
                @"source_files": [unique array]
            }];
        }

        total_tokens += shard_buf_len;
        shard_index++;
    }

    free(shard_buf);

    // Save manifest
    NSDictionary *manifest = nf_manifest_build(shard_info, processed_files,
                                                tokenizer_path, tok.vocab_size,
                                                total_tokens);
    if (!nf_manifest_save(manifest, manifest_path)) {
        nf_emit_error("cannot write manifest", 24);
        nf_tokenizer_free(&tok);
        return 1;
    }
    fprintf(stderr, "Saved manifest: %s (%d tokens, %d shards)\n",
            manifest_path, total_tokens, shard_index);

    // Audit log
    @autoreleasepool {
        char details[NF_AUDIT_DETAIL_MAX];
        snprintf(details, sizeof(details),
                 "\"source\":\"%s\",\"output\":\"%s\","
                 "\"new_files\":%d,\"skipped\":%d,"
                 "\"total_tokens\":%d,\"shards\":%d",
                 source_dir, output_dir, new_files, skipped,
                 total_tokens, shard_index);
        nf_audit_log("ingest", details);
    }

    nf_emit_ingest_done(new_files, skipped, total_tokens, shard_index, manifest_path);
    nf_tokenizer_free(&tok);
    return 0;
}

// ===== Models command =====
static int nf_cmd_models(int argc, char **argv) {
    bool json_output = false;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0) json_output = true;
    }

    if (json_output) {
        nf_emit_model_list();
    } else {
        nf_models_list();
    }
    return 0;
}

// ===== Download command =====
static int nf_cmd_download(int argc, char **argv) {
    const char *model_name = NULL;
    const char *output_dir = NULL;
    const char *hf_token = NULL;

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i+1 < argc) model_name = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i+1 < argc) output_dir = argv[++i];
        else if (strcmp(argv[i], "--token") == 0 && i+1 < argc) hf_token = argv[++i];
    }

    if (!model_name) {
        fprintf(stderr, "Error: --model is required\n");
        fprintf(stderr, "Usage: neuralforge download --model <name> --output <dir>\n");
        fprintf(stderr, "\nAvailable models:\n");
        for (int i = 0; i < nf_model_registry_count; i++)
            fprintf(stderr, "  %s\n", nf_model_registry[i].name);
        return 1;
    }

    if (!output_dir) {
        fprintf(stderr, "Error: --output is required\n");
        return 1;
    }

    // Look up model in registry
    const NFModelEntry *entry = nf_model_find(model_name);
    const char *repo_id = NULL;

    if (entry) {
        repo_id = entry->repo_id;
        fprintf(stderr, "Model: %s (%s)\n", entry->display_name, entry->repo_id);
        fprintf(stderr, "  Architecture: %s\n", entry->architecture);
        fprintf(stderr, "  Parameters: %.0fM\n", entry->params_millions);
        fprintf(stderr, "  Dimensions: dim=%d, hidden=%d, layers=%d\n",
                entry->dim, entry->hidden_dim, entry->n_layers);

        if (entry->gated && !hf_token) {
            fprintf(stderr, "Error: This model requires a HuggingFace token.\n");
            fprintf(stderr, "Get one from https://huggingface.co/settings/tokens\n");
            fprintf(stderr, "Usage: neuralforge download --model %s --output <dir> --token <TOKEN>\n",
                    model_name);
            nf_emit_error("HuggingFace token required for gated model", 1);
            return 1;
        }
    } else {
        // Treat model_name as a HuggingFace repo ID directly
        repo_id = model_name;
        fprintf(stderr, "Model '%s' not in registry, treating as HuggingFace repo ID.\n", model_name);
    }

    // Check Python dependencies
    fprintf(stderr, "\nChecking Python dependencies...\n");
    nf_emit_download_progress(model_name, "checking_deps", 0);

    if (!nf_check_python_deps()) {
        fprintf(stderr, "Error: Required Python packages not found.\n");
        fprintf(stderr, "Install with: pip install safetensors transformers torch\n");
        nf_emit_error("Python dependencies missing: pip install safetensors transformers torch", 1);
        return 1;
    }
    fprintf(stderr, "  Python dependencies OK\n");

    // Find converter script
    NSString *scriptPath = nf_converter_script_path();
    if (!scriptPath) {
        fprintf(stderr, "Error: Cannot find converters/convert_hf.py\n");
        nf_emit_error("converter script not found", 1);
        return 1;
    }
    fprintf(stderr, "  Converter: %s\n", [scriptPath UTF8String]);

    // Run the converter
    @autoreleasepool {
        nf_emit_download_progress(model_name, "downloading", 10);
        fprintf(stderr, "\nStarting download and conversion...\n");

        NSMutableArray *args = [NSMutableArray arrayWithObjects:
            scriptPath,
            @"--model", [NSString stringWithUTF8String:repo_id],
            @"--output", [NSString stringWithUTF8String:output_dir],
            nil];

        if (hf_token) {
            [args addObject:@"--token"];
            [args addObject:[NSString stringWithUTF8String:hf_token]];
        }

        NSTask *task = [[NSTask alloc] init];
        task.executableURL = [NSURL fileURLWithPath:@"/usr/bin/python3"];
        task.arguments = args;

        // Pipe stdout/stderr to our stderr for progress display
        NSPipe *outPipe = [NSPipe pipe];
        task.standardOutput = outPipe;
        task.standardError = outPipe;

        NSError *err = nil;
        [task launchAndReturnError:&err];
        if (err) {
            fprintf(stderr, "Error launching converter: %s\n",
                    [[err localizedDescription] UTF8String]);
            nf_emit_error("failed to launch converter", 1);
            return 1;
        }

        nf_emit_download_progress(model_name, "converting", 50);

        // Read output and display it
        NSFileHandle *handle = outPipe.fileHandleForReading;
        NSData *data;
        while ((data = [handle availableData]) && data.length > 0) {
            NSString *line = [[NSString alloc] initWithData:data
                                                   encoding:NSUTF8StringEncoding];
            if (line) fprintf(stderr, "%s", [line UTF8String]);
        }

        [task waitUntilExit];

        if (task.terminationStatus != 0) {
            fprintf(stderr, "\nConversion failed (exit code %d)\n", task.terminationStatus);
            nf_emit_download_done(model_name, NULL, NULL, false);
            return 1;
        }

        // Check output files
        NSString *outDir = [NSString stringWithUTF8String:output_dir];
        NSString *modelPath = [outDir stringByAppendingPathComponent:@"model.bin"];
        NSString *tokPath = [outDir stringByAppendingPathComponent:@"tokenizer.bin"];
        NSString *cardPath = [outDir stringByAppendingPathComponent:@"model_card.json"];

        bool has_model = [[NSFileManager defaultManager] fileExistsAtPath:modelPath];
        bool has_tok = [[NSFileManager defaultManager] fileExistsAtPath:tokPath];

        if (!has_model) {
            fprintf(stderr, "Error: model.bin not found in output directory\n");
            nf_emit_download_done(model_name, NULL, NULL, false);
            return 1;
        }

        nf_emit_download_progress(model_name, "complete", 100);
        nf_emit_download_done(model_name,
                              [modelPath UTF8String],
                              has_tok ? [tokPath UTF8String] : NULL,
                              true);

        // Emit model card if available
        if ([[NSFileManager defaultManager] fileExistsAtPath:cardPath]) {
            nf_emit_model_card([cardPath UTF8String]);
        }

        fprintf(stderr, "\nDownload and conversion complete!\n");
        fprintf(stderr, "  Model:     %s\n", [modelPath UTF8String]);
        if (has_tok)
            fprintf(stderr, "  Tokenizer: %s\n", [tokPath UTF8String]);

        // Audit log
        char details[512];
        snprintf(details, sizeof(details),
                 "model=%s repo=%s output=%s",
                 model_name, repo_id, output_dir);
        nf_audit_log("download", details);
    }

    return 0;
}

// ===== Main entry point =====
static void print_usage(void) {
    fprintf(stderr,
        "NeuralForge — On-device AI training via Apple Neural Engine\n"
        "\n"
        "Usage:\n"
        "  neuralforge train     [options]   Train a model\n"
        "  neuralforge generate  [options]   Generate text from a model\n"
        "  neuralforge tokenize  [options]   Tokenize text to binary tokens\n"
        "  neuralforge ingest    [options]   Ingest documents into token shards\n"
        "  neuralforge export    [options]   Export checkpoint to model format\n"
        "  neuralforge info      [options]   Show model info\n"
        "  neuralforge models    [--json]    List available pre-trained models\n"
        "  neuralforge download  [options]   Download and convert a HuggingFace model\n"
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
        "  --warmup N        Warmup steps (linear ramp, default: 0)\n"
        "  --lr-min F        Minimum LR after decay (default: 1e-5)\n"
        "  --lr-schedule S   LR schedule: none, cosine (default: none)\n"
        "  --val-data PATH   Validation data file (optional)\n"
        "  --val-every N     Validate every N optimizer steps (default: 0=off)\n"
        "  --val-batches N   Number of val batches to average (default: 10)\n"
        "  --shuffle         Shuffle data positions via xorshift32\n"
        "  --lora-rank N     LoRA rank (0=full fine-tune, 4-64 typical, default: 0)\n"
        "  --lora-alpha F    LoRA scaling factor (default: 16)\n"
        "  --lora-targets N  LoRA target bitmask: 8=Wo (default: 8)\n"
        "\n"
        "Tokenize options:\n"
        "  --input PATH      Input text file (required)\n"
        "  --output PATH     Output binary token file (default: tokens.bin)\n"
        "  --tokenizer PATH  Tokenizer model file (default: tokenizer.bin)\n"
        "\n"
        "Ingest options:\n"
        "  --source DIR      Source directory to scan for documents (required)\n"
        "  --output DIR      Output directory for token shards (required)\n"
        "  --tokenizer PATH  Tokenizer model file (default: tokenizer.bin)\n"
        "  --max-shard-mb N  Max shard size in MB (default: 50)\n"
        "  --manifest PATH   Path to manifest.json (default: output/manifest.json)\n"
        "  --incremental     Only process new/modified files since last run\n"
        "\n"
        "Export options:\n"
        "  --ckpt PATH       Checkpoint to export (required)\n"
        "  --format FMT      Output format: llama2c, gguf (default: llama2c)\n"
        "  --output PATH     Output file path (required)\n"
        "\n"
        "Generate options:\n"
        "  --model PATH      Model weights file (default: stories110M.bin)\n"
        "  --tokenizer PATH  Tokenizer file (default: tokenizer.bin)\n"
        "  --prompt TEXT     Input prompt text\n"
        "  --temperature F   Sampling temperature (default: 0.8)\n"
        "  --top-p F         Nucleus sampling threshold (default: 0.9)\n"
        "  --max-tokens N    Maximum tokens to generate (default: 256)\n"
        "  --seed N          Random seed (default: 42)\n"
        "\n"
        "Models options:\n"
        "  --json            Output model list as JSON (for app consumption)\n"
        "\n"
        "Download options:\n"
        "  --model NAME      Model name from registry or HuggingFace repo ID (required)\n"
        "  --output DIR      Output directory for converted files (required)\n"
        "  --token TOKEN     HuggingFace API token (for gated models like Llama 2)\n"
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
        model_config_defaults();  // Stories110M defaults until model/ckpt loaded
        ane_init();
        mach_timebase_info(&g_tb);
        nf_audit_init();

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

        if (strcmp(cmd, "ingest") == 0) {
            return nf_cmd_ingest(argc - 2, argv + 2);
        }

        if (strcmp(cmd, "models") == 0) {
            return nf_cmd_models(argc - 2, argv + 2);
        }

        if (strcmp(cmd, "download") == 0) {
            return nf_cmd_download(argc - 2, argv + 2);
        }

        if (strcmp(cmd, "export") == 0) {
            return nf_cmd_export(argc - 2, argv + 2);
        }

        if (strcmp(cmd, "generate") == 0) {
            return nf_cmd_generate(argc - 2, argv + 2);
        }

        if (strcmp(cmd, "benchmark") == 0) {
            return nf_cmd_benchmark(&cfg);
        }

        fprintf(stderr, "Unknown command: %s\n", cmd);
        print_usage();
        return 1;
    }
}
