// progress.h — JSON progress reporting for NeuralForge CLI
// All output to stdout is valid JSON (one object per line)
// Human-readable logs go to stderr only
#pragma once
#include <stdio.h>
#include <time.h>
#include <string.h>

// JSON-escape a string: handles \, ", \n, \r, \t, and control chars
// Returns bytes written (excluding NUL). Output is always NUL-terminated.
static size_t nf_json_escape(const char *input, char *output, size_t max_len) {
    if (!input || !output || max_len == 0) { if (output && max_len) output[0] = '\0'; return 0; }
    size_t o = 0;
    for (const char *p = input; *p && o < max_len - 1; p++) {
        unsigned char ch = (unsigned char)*p;
        if (ch == '"' || ch == '\\') {
            if (o + 2 > max_len - 1) break;
            output[o++] = '\\'; output[o++] = (char)ch;
        } else if (ch == '\n') {
            if (o + 2 > max_len - 1) break;
            output[o++] = '\\'; output[o++] = 'n';
        } else if (ch == '\r') {
            if (o + 2 > max_len - 1) break;
            output[o++] = '\\'; output[o++] = 'r';
        } else if (ch == '\t') {
            if (o + 2 > max_len - 1) break;
            output[o++] = '\\'; output[o++] = 't';
        } else if (ch < 0x20) {
            // Control characters → \u00XX
            if (o + 6 > max_len - 1) break;
            snprintf(output + o, 7, "\\u%04x", ch);
            o += 6;
        } else {
            output[o++] = (char)ch;
        }
    }
    output[o] = '\0';
    return o;
}

static long nf_timestamp(void) {
    return (long)time(NULL);
}

static void nf_emit_init(int params, int layers, int dim, int hidden,
                          int heads, int seq, int vocab) {
    fprintf(stdout, "{\"type\":\"init\",\"params\":%d,\"layers\":%d,"
            "\"dim\":%d,\"hidden\":%d,\"heads\":%d,\"seq\":%d,\"vocab\":%d,"
            "\"timestamp\":%ld}\n",
            params, layers, dim, hidden, heads, seq, vocab, nf_timestamp());
    fflush(stdout);
}

static void nf_emit_step(int step, int total, float loss, float lr,
                          double ms, double tflops_ane, double tflops_total) {
    fprintf(stdout, "{\"type\":\"step\",\"step\":%d,\"total\":%d,"
            "\"loss\":%.6f,\"lr\":%.6e,\"ms\":%.2f,"
            "\"tflops_ane\":%.3f,\"tflops_total\":%.3f}\n",
            step, total, loss, lr, ms, tflops_ane, tflops_total);
    fflush(stdout);
}

static void nf_emit_batch(int batch, int step, float avg_loss, float best_loss,
                           double compile_ms, double train_ms, int compiles) {
    fprintf(stdout, "{\"type\":\"batch\",\"batch\":%d,\"step\":%d,"
            "\"avg_loss\":%.6f,\"best_loss\":%.6f,"
            "\"compile_ms\":%.1f,\"train_ms\":%.1f,\"compiles\":%d}\n",
            batch, step, avg_loss, best_loss, compile_ms, train_ms, compiles);
    fflush(stdout);
}

static void nf_emit_checkpoint(const char *path, int step, float loss) {
    char escaped[PATH_MAX * 2];
    nf_json_escape(path, escaped, sizeof(escaped));
    fprintf(stdout, "{\"type\":\"checkpoint\",\"path\":\"%s\","
            "\"step\":%d,\"loss\":%.6f}\n", escaped, step, loss);
    fflush(stdout);
}

static void nf_emit_restart(int step, int compiles, const char *reason) {
    char escaped[256];
    nf_json_escape(reason, escaped, sizeof(escaped));
    fprintf(stdout, "{\"type\":\"restart\",\"step\":%d,\"compiles\":%d,"
            "\"reason\":\"%s\"}\n", step, compiles, escaped);
    fflush(stdout);
}

static void nf_emit_done(int total_steps, float final_loss, double total_time_s,
                          double tflops_ane, double tflops_total) {
    fprintf(stdout, "{\"type\":\"done\",\"total_steps\":%d,"
            "\"final_loss\":%.6f,\"total_time_s\":%.2f,"
            "\"tflops_ane\":%.3f,\"tflops_total\":%.3f}\n",
            total_steps, final_loss, total_time_s, tflops_ane, tflops_total);
    fflush(stdout);
}

static void nf_emit_error(const char *message, int code) {
    char escaped[1024];
    nf_json_escape(message, escaped, sizeof(escaped));
    fprintf(stdout, "{\"type\":\"error\",\"message\":\"%s\",\"code\":%d}\n",
            escaped, code);
    fflush(stdout);
}

static void nf_emit_val(int step, float val_loss, int val_batches) {
    fprintf(stdout, "{\"type\":\"val\",\"step\":%d,\"val_loss\":%.6f,"
            "\"val_batches\":%d}\n", step, val_loss, val_batches);
    fflush(stdout);
}

static void nf_emit_token(int token_id, const char *text) {
    char escaped[512];
    nf_json_escape(text, escaped, sizeof(escaped));
    fprintf(stdout, "{\"type\":\"token\",\"token_id\":%d,\"text\":\"%s\"}\n",
            token_id, escaped);
    fflush(stdout);
}

static void nf_emit_generate_done(int tokens, double total_ms) {
    fprintf(stdout, "{\"type\":\"generate_done\",\"tokens\":%d,\"total_ms\":%.1f}\n",
            tokens, total_ms);
    fflush(stdout);
}

static void nf_emit_lora_info(int lora_rank, float lora_alpha, int lora_targets, int lora_params) {
    fprintf(stdout, "{\"type\":\"info\",\"key\":\"lora\","
            "\"value\":{\"rank\":%d,\"alpha\":%.1f,\"targets\":%d,\"params\":%d}}\n",
            lora_rank, lora_alpha, lora_targets, lora_params);
    fflush(stdout);
}

static void nf_emit_info(const char *key, const char *value) {
    char ek[256];
    nf_json_escape(key, ek, sizeof(ek));
    fprintf(stdout, "{\"type\":\"info\",\"key\":\"%s\",\"value\":%s}\n",
            ek, value);
    fflush(stdout);
}

static void nf_emit_ingest_file(const char *filename, int tokens) {
    char escaped[PATH_MAX * 2];
    nf_json_escape(filename, escaped, sizeof(escaped));
    fprintf(stdout, "{\"type\":\"ingest_file\",\"file\":\"%s\",\"tokens\":%d}\n",
            escaped, tokens);
    fflush(stdout);
}

static void nf_emit_ingest_done(int new_files, int skipped, int total_tokens,
                                 int shards, const char *manifest_path) {
    char escaped[PATH_MAX * 2];
    nf_json_escape(manifest_path, escaped, sizeof(escaped));
    fprintf(stdout, "{\"type\":\"ingest_done\",\"new_files\":%d,\"skipped\":%d,"
            "\"total_tokens\":%d,\"shards\":%d,\"manifest\":\"%s\"}\n",
            new_files, skipped, total_tokens, shards, escaped);
    fflush(stdout);
}

static void nf_emit_download_progress(const char *model_name, const char *status,
                                       int percent) {
    char name_esc[256], status_esc[256];
    nf_json_escape(model_name, name_esc, sizeof(name_esc));
    nf_json_escape(status, status_esc, sizeof(status_esc));
    fprintf(stdout, "{\"type\":\"download_progress\",\"model\":\"%s\","
            "\"status\":\"%s\",\"percent\":%d}\n",
            name_esc, status_esc, percent);
    fflush(stdout);
}

static void nf_emit_download_done(const char *model_name, const char *model_path,
                                   const char *tokenizer_path, bool success) {
    char m_esc[256], mp_esc[PATH_MAX * 2], tp_esc[PATH_MAX * 2];
    nf_json_escape(model_name, m_esc, sizeof(m_esc));
    nf_json_escape(model_path ? model_path : "", mp_esc, sizeof(mp_esc));
    nf_json_escape(tokenizer_path ? tokenizer_path : "", tp_esc, sizeof(tp_esc));
    fprintf(stdout, "{\"type\":\"download_done\",\"model\":\"%s\","
            "\"model_path\":\"%s\",\"tokenizer_path\":\"%s\",\"success\":%s}\n",
            m_esc, mp_esc, tp_esc, success ? "true" : "false");
    fflush(stdout);
}
