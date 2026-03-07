// stories_cpu_ops.h — CPU operations: RMSNorm, cross-entropy, Adam, softmax
#pragma once
#include "stories_config.h"


static void rmsnorm(float *out, const float *x, const float *w, int d, int S) {
    float *rms_tmp = (float*)malloc(S * sizeof(float));
    float *ss = (float*)calloc(S, sizeof(float));
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, x+i*S, 1, rms_tmp, 1, (vDSP_Length)S);
        vDSP_vadd(rms_tmp, 1, ss, 1, ss, 1, (vDSP_Length)S);
    }
    float invd = 1.0f/d, eps=1e-5f;
    vDSP_vsmsa(ss, 1, &invd, &eps, ss, 1, (vDSP_Length)S);
    int n = S; vvrsqrtf(ss, ss, &n);
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, ss, 1, out+i*S, 1, (vDSP_Length)S);
        vDSP_vsmul(out+i*S, 1, &w[i], out+i*S, 1, (vDSP_Length)S);
    }
    free(ss); free(rms_tmp);
}

static void rmsnorm_bwd(float *dx, float *dw, const float *dy, const float *x, const float *w, int d, int S) {
    float *rms_tmp = (float*)malloc(S * sizeof(float));
    float *ss = (float*)calloc(S, sizeof(float));
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, x+i*S, 1, rms_tmp, 1, (vDSP_Length)S);
        vDSP_vadd(rms_tmp, 1, ss, 1, ss, 1, (vDSP_Length)S);
    }
    float invd = 1.0f/d, eps=1e-5f;
    vDSP_vsmsa(ss, 1, &invd, &eps, ss, 1, (vDSP_Length)S);
    float *rrms = (float*)malloc(S*4);
    int n = S; vvrsqrtf(rrms, ss, &n);
    float *dot = (float*)calloc(S, sizeof(float));
    for (int i=0; i<d; i++) {
        vDSP_vmul(dy+i*S, 1, x+i*S, 1, rms_tmp, 1, (vDSP_Length)S);
        vDSP_vsma(rms_tmp, 1, &w[i], dot, 1, dot, 1, (vDSP_Length)S);
    }
    vDSP_vmul(rrms, 1, rrms, 1, ss, 1, (vDSP_Length)S);
    vDSP_vsmul(ss, 1, &invd, ss, 1, (vDSP_Length)S);
    vDSP_vmul(dot, 1, ss, 1, dot, 1, (vDSP_Length)S);
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, dot, 1, rms_tmp, 1, (vDSP_Length)S);
        vDSP_vsub(rms_tmp, 1, dy+i*S, 1, rms_tmp, 1, (vDSP_Length)S);
        vDSP_vmul(rms_tmp, 1, rrms, 1, rms_tmp, 1, (vDSP_Length)S);
        vDSP_vsmul(rms_tmp, 1, &w[i], dx+i*S, 1, (vDSP_Length)S);
        vDSP_vmul(dy+i*S, 1, x+i*S, 1, rms_tmp, 1, (vDSP_Length)S);
        vDSP_vmul(rms_tmp, 1, rrms, 1, rms_tmp, 1, (vDSP_Length)S);
        float s; vDSP_sve(rms_tmp, 1, &s, (vDSP_Length)S);
        dw[i] += s;
    }
    free(ss); free(rrms); free(dot); free(rms_tmp);
}

static void adam_update(float *w, const float *g, AdamState *s, int t, float lr, float b1, float b2, float eps) {
    float bc1 = 1.0f - powf(b1, t), bc2 = 1.0f - powf(b2, t);
    for (size_t i=0; i<s->n; i++) {
        s->m[i] = b1*s->m[i] + (1-b1)*g[i];
        s->v[i] = b2*s->v[i] + (1-b2)*g[i]*g[i];
        float mh = s->m[i]/bc1, vh = s->v[i]/bc2;
        w[i] -= lr * mh / (sqrtf(vh) + eps);
    }
}

// Cross-entropy loss + gradient for logits (column-major: [VOCAB, SEQ])
// logits[v*SEQ+t] = logit for vocab v, position t
// targets[t] = target token id for position t
// Returns mean CE loss, writes dlogits = softmax(logits) - one_hot(targets)
// Data is column-major [V, S], but we process per-column (stride=1 within col is v*S+t, stride between v's is S)
// For vDSP: transpose to row-major scratch [S, V] to vectorize softmax per position
static float cross_entropy_loss(float *dlogits, const float *logits, const uint16_t *targets, int V, int S) {
    // Work in transposed layout [S, V] where each row is one position's logits (contiguous)
    float *buf = (float*)malloc(S * V * 4);
    // Transpose [V,S] → [S,V]: buf[t*V+v] = logits[v*S+t]
    vDSP_mtrans(logits, 1, buf, 1, (vDSP_Length)S, (vDSP_Length)V);

    float total_loss = 0;
    float invS = 1.0f / S;
    for (int t = 0; t < S; t++) {
        float *row = buf + t * V;
        // max
        float maxv;
        vDSP_maxv(row, 1, &maxv, (vDSP_Length)V);
        // row -= maxv
        float neg_max = -maxv;
        vDSP_vsadd(row, 1, &neg_max, row, 1, (vDSP_Length)V);
        // exp in-place
        int n = V;
        vvexpf(row, row, &n);
        // sum
        float sum;
        vDSP_sve(row, 1, &sum, (vDSP_Length)V);
        // normalize
        float inv_sum = 1.0f / sum;
        vDSP_vsmul(row, 1, &inv_sum, row, 1, (vDSP_Length)V);
        // loss
        int tgt = targets[t];
        if (tgt < 0 || tgt >= V) { fprintf(stderr, "WARN: target token %d out of vocab range [0,%d), skipping\n", tgt, V); continue; }
        total_loss -= logf(row[tgt] + 1e-10f);
        // gradient: softmax - one_hot, then /S
        row[tgt] -= 1.0f;
        vDSP_vsmul(row, 1, &invS, row, 1, (vDSP_Length)V);
    }
    // Transpose back [S,V] → [V,S]
    vDSP_mtrans(buf, 1, dlogits, 1, (vDSP_Length)V, (vDSP_Length)S);
    free(buf);
    return total_loss / S;
}

// Embedding lookup: token_ids → x [dim, seq] (channel-first)
// embed is [vocab, dim] row-major (vocab_size rows, dim cols)
static void embed_lookup(float *x, const float *embed, const uint16_t *tokens, int dim, int seq, int vocab) {
    for (int t = 0; t < seq; t++) {
        int tok = tokens[t];
        if (tok >= vocab) {
            fprintf(stderr, "WARN: token %d out of range [0,%d)\n", tok, vocab);
            for (int d = 0; d < dim; d++) x[d*seq + t] = 0.0f;  // zero out to prevent uninitialized propagation
            continue;
        }
        for (int d = 0; d < dim; d++) {
            x[d*seq + t] = embed[tok*dim + d];
        }
    }
}

// LoRA forward: output += (B @ A @ input) * scale
// input:  [in_dim, seq]  column-major
// output: [out_dim, seq] column-major (modified in-place)
// A: [rank, in_dim] row-major
// B: [out_dim, rank] row-major
// scale = alpha / rank
static void lora_forward(float *output, const float *input,
                          const float *A, const float *B,
                          int in_dim, int out_dim, int rank, float scale, int seq) {
    float *temp = (float*)malloc((size_t)rank * seq * 4);
    // temp[rank, seq] = A[rank, in_dim] @ input[in_dim, seq]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rank, seq, in_dim, 1.0f, A, in_dim, input, seq, 0.0f, temp, seq);
    // output[out_dim, seq] += B[out_dim, rank] @ temp[rank, seq] * scale
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                out_dim, seq, rank, scale, B, rank, temp, seq, 1.0f, output, seq);
    free(temp);
}

// LoRA backward: compute grad_A, grad_B (accumulated)
// grad_output: [out_dim, seq] — gradient w.r.t. layer output
// input: [in_dim, seq] — saved activation (e.g. attn_out for Wo)
// Returns: grad_A [rank, in_dim], grad_B [out_dim, rank] (accumulated)
static void lora_backward(float *grad_A, float *grad_B,
                            const float *grad_output, const float *input,
                            const float *A, const float *B,
                            int in_dim, int out_dim, int rank, float scale, int seq) {
    // Forward was: temp = A @ input; delta = B @ temp * scale
    // Recompute temp for backward
    float *temp = (float*)malloc((size_t)rank * seq * 4);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rank, seq, in_dim, 1.0f, A, in_dim, input, seq, 0.0f, temp, seq);

    // grad_B[out_dim, rank] += scale * grad_output[out_dim, seq] @ temp^T[seq, rank]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                out_dim, rank, seq, scale, grad_output, seq, temp, seq, 1.0f, grad_B, rank);

    // d_temp[rank, seq] = scale * B^T[rank, out_dim] @ grad_output[out_dim, seq]
    float *d_temp = (float*)malloc((size_t)rank * seq * 4);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                rank, seq, out_dim, scale, B, rank, grad_output, seq, 0.0f, d_temp, seq);

    // grad_A[rank, in_dim] += d_temp[rank, seq] @ input^T[seq, in_dim]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                rank, in_dim, seq, 1.0f, d_temp, seq, input, seq, 1.0f, grad_A, in_dim);

    free(temp);
    free(d_temp);
}

// Embedding backward: accumulate dE[tok] += dx[:,t] for each position
static void embed_backward(float *d_embed, const float *dx, const uint16_t *tokens, int dim, int seq, int vocab) {
    for (int t = 0; t < seq; t++) {
        int tok = tokens[t];
        if (tok < 0 || tok >= vocab) { continue; }
        for (int d = 0; d < dim; d++) {
            d_embed[tok*dim + d] += dx[d*seq + t];
        }
    }
}
