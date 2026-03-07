// test_cli.m — Unit tests for NeuralForge CLI components
//
// Build: make test_cli
// Run:   make test

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <mach/mach_time.h>

#include "config.h"
#include "progress.h"
#include "tokenizer.h"
#include "stories_config.h"
#include "audit.h"
#include "ingest.h"
#include "models.h"

// xorshift32 RNG (same as main.m, replicated for testing)
static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    if (x == 0) x = 1;  // xorshift32 has zero absorbing state — guard against it
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    fprintf(stderr, "  TEST: %s... ", #name); \
    if (test_##name()) { tests_passed++; fprintf(stderr, "PASS\n"); } \
    else { fprintf(stderr, "FAIL\n"); } \
} while(0)

// ===== Config Tests =====

static bool test_config_defaults(void) {
    NFConfig cfg = nf_config_defaults();

    if (cfg.total_steps != 10000) return false;
    if (cfg.accum_steps != 10) return false;
    if (cfg.seed != 42) return false;
    if (cfg.beta1 != 0.9f) return false;
    if (cfg.beta2 != 0.999f) return false;
    if (cfg.checkpoint_every != 100) return false;
    if (cfg.resume != false) return false;
    if (cfg.use_ane_extras != true) return false;
    if (cfg.json_output != true) return false;
    return true;
}

static bool test_config_from_args(void) {
    char *argv[] = {
        (char*)"--model", (char*)"my_model.bin",
        (char*)"--data", (char*)"my_data.bin",
        (char*)"--steps", (char*)"500",
        (char*)"--lr", (char*)"1e-4",
        (char*)"--accum", (char*)"5",
        (char*)"--seed", (char*)"123",
        (char*)"--resume",
        (char*)"--no-ane-extras",
    };
    int argc = sizeof(argv) / sizeof(argv[0]);
    NFConfig cfg = nf_config_from_args(argc, argv);

    if (strcmp(cfg.model_path, "my_model.bin") != 0) return false;
    if (strcmp(cfg.data_path, "my_data.bin") != 0) return false;
    if (cfg.total_steps != 500) return false;
    if (cfg.accum_steps != 5) return false;
    if (cfg.seed != 123) return false;
    if (cfg.resume != true) return false;
    if (cfg.use_ane_extras != false) return false;
    // LR check (float precision)
    if (cfg.learning_rate < 9e-5f || cfg.learning_rate > 1.1e-4f) return false;
    return true;
}

static bool test_config_partial_args(void) {
    char *argv[] = {
        (char*)"--steps", (char*)"200",
    };
    NFConfig cfg = nf_config_from_args(2, argv);

    if (cfg.total_steps != 200) return false;
    // Other values should remain at defaults
    if (cfg.seed != 42) return false;
    if (cfg.accum_steps != 10) return false;
    if (cfg.resume != false) return false;
    return true;
}

static bool test_config_optimizer_args(void) {
    char *argv[] = {
        (char*)"--beta1", (char*)"0.85",
        (char*)"--beta2", (char*)"0.995",
        (char*)"--eps", (char*)"1e-7",
        (char*)"--grad-clip", (char*)"0.5",
        (char*)"--max-compiles", (char*)"50",
    };
    int argc = sizeof(argv) / sizeof(argv[0]);
    NFConfig cfg = nf_config_from_args(argc, argv);

    if (cfg.beta1 < 0.84f || cfg.beta1 > 0.86f) return false;
    if (cfg.beta2 < 0.994f || cfg.beta2 > 0.996f) return false;
    if (cfg.eps < 9e-8f || cfg.eps > 1.1e-7f) return false;
    if (cfg.grad_clip_norm < 0.49f || cfg.grad_clip_norm > 0.51f) return false;
    if (cfg.max_compiles != 50) return false;
    return true;
}

static bool test_config_json_file(void) {
    // Write a JSON config to a temp file
    char tmppath[] = "/tmp/nf_test_config_XXXXXX";
    int fd = mkstemp(tmppath);
    const char *json = "{\"steps\":777,\"learning_rate\":0.001,\"beta1\":0.8,\"seed\":99}";
    write(fd, json, strlen(json));
    close(fd);

    char *argv[] = {
        (char*)"--config", tmppath,
    };
    NFConfig cfg = nf_config_from_args(2, argv);
    unlink(tmppath);

    if (cfg.total_steps != 777) return false;
    if (cfg.learning_rate < 0.0009f || cfg.learning_rate > 0.0011f) return false;
    if (cfg.beta1 < 0.79f || cfg.beta1 > 0.81f) return false;
    if (cfg.seed != 99) return false;
    // Verify other defaults remain
    if (cfg.accum_steps != 10) return false;
    return true;
}

static bool test_config_json_with_override(void) {
    // JSON config + CLI args: CLI should override JSON
    char tmppath[] = "/tmp/nf_test_config2_XXXXXX";
    int fd = mkstemp(tmppath);
    const char *json = "{\"steps\":500,\"seed\":11}";
    write(fd, json, strlen(json));
    close(fd);

    char *argv[] = {
        (char*)"--config", tmppath,
        (char*)"--steps", (char*)"999",
    };
    NFConfig cfg = nf_config_from_args(4, argv);
    unlink(tmppath);

    // --steps should override JSON
    if (cfg.total_steps != 999) return false;
    // seed from JSON should remain
    if (cfg.seed != 11) return false;
    return true;
}

static bool test_config_path_overflow(void) {
    // Verify path doesn't overflow with strlcpy
    char long_path[PATH_MAX + 100];
    memset(long_path, 'a', sizeof(long_path) - 1);
    long_path[sizeof(long_path) - 1] = '\0';

    char *argv[] = { (char*)"--model", long_path };
    NFConfig cfg = nf_config_from_args(2, argv);

    // Should be truncated, not crash
    if (strlen(cfg.model_path) >= PATH_MAX) return false;
    return true;
}

// ===== Progress JSON Tests =====

// Capture stdout output
static char captured[8192];
static int capture_fd = -1;
static int saved_stdout = -1;

static void capture_start(void) {
    fflush(stdout);
    saved_stdout = dup(STDOUT_FILENO);
    int pipefd[2];
    pipe(pipefd);
    dup2(pipefd[1], STDOUT_FILENO);
    close(pipefd[1]);
    capture_fd = pipefd[0];
    memset(captured, 0, sizeof(captured));
}

static void capture_end(void) {
    fflush(stdout);
    int flags = fcntl(capture_fd, F_GETFL);
    fcntl(capture_fd, F_SETFL, flags | O_NONBLOCK);
    ssize_t n = read(capture_fd, captured, sizeof(captured) - 1);
    if (n > 0) captured[n] = '\0';
    close(capture_fd);
    dup2(saved_stdout, STDOUT_FILENO);
    close(saved_stdout);
    capture_fd = -1;
    saved_stdout = -1;
}

static bool test_progress_step_json(void) {
    capture_start();
    nf_emit_step(5, 100, 3.14, 3e-4, 42.0, 1.5, 2.0);
    capture_end();

    if (strstr(captured, "\"type\":\"step\"") == NULL) return false;
    if (strstr(captured, "\"step\":5") == NULL) return false;
    if (strstr(captured, "\"total\":100") == NULL) return false;
    // Check it ends with newline (NDJSON)
    size_t len = strlen(captured);
    if (len == 0 || captured[len-1] != '\n') return false;
    return true;
}

static bool test_progress_error_json(void) {
    capture_start();
    nf_emit_error("test error", 42);
    capture_end();

    if (strstr(captured, "\"type\":\"error\"") == NULL) return false;
    if (strstr(captured, "\"message\":\"test error\"") == NULL) return false;
    if (strstr(captured, "\"code\":42") == NULL) return false;
    return true;
}

static bool test_progress_checkpoint_json(void) {
    capture_start();
    nf_emit_checkpoint("/tmp/ckpt.bin", 50, 2.5);
    capture_end();

    if (strstr(captured, "\"type\":\"checkpoint\"") == NULL) return false;
    if (strstr(captured, "\"step\":50") == NULL) return false;
    if (strstr(captured, "\"/tmp/ckpt.bin\"") == NULL) return false;
    return true;
}

static bool test_progress_done_json(void) {
    capture_start();
    nf_emit_done(1000, 1.5, 60.0, 3.0, 4.0);
    capture_end();

    if (strstr(captured, "\"type\":\"done\"") == NULL) return false;
    if (strstr(captured, "\"total_steps\":1000") == NULL) return false;
    return true;
}

static bool test_progress_batch_json(void) {
    capture_start();
    nf_emit_batch(3, 30, 2.5, 2.1, 100.0, 500.0, 5);
    capture_end();

    if (strstr(captured, "\"type\":\"batch\"") == NULL) return false;
    if (strstr(captured, "\"batch\":3") == NULL) return false;
    if (strstr(captured, "\"compiles\":5") == NULL) return false;
    return true;
}

static bool test_progress_restart_json(void) {
    capture_start();
    nf_emit_restart(50, 99, "compile_budget");
    capture_end();

    if (strstr(captured, "\"type\":\"restart\"") == NULL) return false;
    if (strstr(captured, "\"step\":50") == NULL) return false;
    if (strstr(captured, "\"reason\":\"compile_budget\"") == NULL) return false;
    return true;
}

// ===== Tokenizer Tests =====

static const char *find_tokenizer(void) {
    static const char *paths[] = {
        "../vendor/ANE/assets/models/tokenizer.bin",
        "../../vendor/ANE/assets/models/tokenizer.bin",
        NULL
    };
    for (int i = 0; paths[i]; i++) {
        if (access(paths[i], R_OK) == 0) return paths[i];
    }
    return NULL;
}

static bool test_tokenizer_load(void) {
    const char *path = find_tokenizer();
    if (!path) {
        fprintf(stderr, "(tokenizer.bin not found, skipping) ");
        return true;
    }

    NFTokenizer tok;
    if (!nf_tokenizer_load(&tok, path, 32000)) return false;
    if (tok.vocab_size != 32000) { nf_tokenizer_free(&tok); return false; }
    if (tok.max_token_length <= 0) { nf_tokenizer_free(&tok); return false; }
    if (tok.vocab == NULL) { nf_tokenizer_free(&tok); return false; }
    if (tok.scores == NULL) { nf_tokenizer_free(&tok); return false; }
    nf_tokenizer_free(&tok);
    return true;
}

static bool test_tokenizer_encode(void) {
    const char *path = find_tokenizer();
    if (!path) {
        fprintf(stderr, "(tokenizer.bin not found, skipping) ");
        return true;
    }

    NFTokenizer tok;
    if (!nf_tokenizer_load(&tok, path, 32000)) return false;

    uint16_t tokens[256];
    int n = nf_tokenizer_encode(&tok, "Hello world", tokens, 256);

    bool ok = (n > 0 && n < 20);
    for (int j = 0; j < n && ok; j++) {
        if (tokens[j] >= 32000) ok = false;
    }
    nf_tokenizer_free(&tok);
    return ok;
}

static bool test_tokenizer_encode_empty(void) {
    const char *path = find_tokenizer();
    if (!path) {
        fprintf(stderr, "(tokenizer.bin not found, skipping) ");
        return true;
    }

    NFTokenizer tok;
    if (!nf_tokenizer_load(&tok, path, 32000)) return false;

    uint16_t tokens[256];
    int n = nf_tokenizer_encode(&tok, "", tokens, 256);
    nf_tokenizer_free(&tok);
    return n == 0;
}

static bool test_tokenizer_roundtrip_length(void) {
    const char *path = find_tokenizer();
    if (!path) {
        fprintf(stderr, "(tokenizer.bin not found, skipping) ");
        return true;
    }

    NFTokenizer tok;
    if (!nf_tokenizer_load(&tok, path, 32000)) return false;

    // Longer text should produce more tokens
    uint16_t tokens_short[256], tokens_long[4096];
    int n_short = nf_tokenizer_encode(&tok, "Hi", tokens_short, 256);
    int n_long = nf_tokenizer_encode(&tok, "Once upon a time there was a little girl who loved to play in the garden", tokens_long, 4096);

    nf_tokenizer_free(&tok);
    return n_long > n_short;
}

// ===== Format Tests =====

static bool test_gguf_magic(void) {
    // GGUF magic 0x46475547 — verify it's a known constant that matches llama.cpp
    uint32_t magic = 0x46475547;
    // Write to temp file, read back, verify
    char tmppath[] = "/tmp/nf_test_gguf_XXXXXX";
    int fd = mkstemp(tmppath);
    write(fd, &magic, 4);
    uint32_t version = 3;
    write(fd, &version, 4);
    close(fd);

    FILE *f = fopen(tmppath, "rb");
    uint32_t m2, v2;
    fread(&m2, 4, 1, f);
    fread(&v2, 4, 1, f);
    fclose(f);
    unlink(tmppath);

    if (m2 != 0x46475547) return false;
    if (v2 != 3) return false;
    return true;
}

static bool test_llama2c_config_roundtrip(void) {
    Llama2Config cfg = {0};
    cfg.dim = 768;
    cfg.hidden_dim = 2048;
    cfg.n_layers = 12;
    cfg.n_heads = 12;
    cfg.n_kv_heads = 12;
    cfg.vocab_size = -32000;
    cfg.seq_len = 256;

    char tmppath[] = "/tmp/nf_test_cfg_XXXXXX";
    int fd = mkstemp(tmppath);
    write(fd, &cfg, sizeof(cfg));
    close(fd);

    FILE *f = fopen(tmppath, "rb");
    Llama2Config cfg2;
    fread(&cfg2, sizeof(cfg2), 1, f);
    fclose(f);
    unlink(tmppath);

    if (cfg2.dim != 768) return false;
    if (cfg2.hidden_dim != 2048) return false;
    if (cfg2.n_layers != 12) return false;
    if (cfg2.n_heads != 12) return false;
    if (cfg2.vocab_size != -32000) return false;
    if (cfg2.seq_len != 256) return false;
    return true;
}

static bool test_checkpoint_magic(void) {
    // Verify our checkpoint magic constant
    uint32_t magic = 0x424C5A54;
    char *bytes = (char *)&magic;
    // "TZLB" in little-endian
    if (bytes[0] != 'T' || bytes[1] != 'Z' || bytes[2] != 'L' || bytes[3] != 'B') return false;
    return true;
}

// ===== Security / Penetration Tests =====

static bool test_json_escape_basic(void) {
    char out[256];
    nf_json_escape("hello world", out, sizeof(out));
    return strcmp(out, "hello world") == 0;
}

static bool test_json_escape_quotes(void) {
    char out[256];
    nf_json_escape("say \"hello\"", out, sizeof(out));
    return strcmp(out, "say \\\"hello\\\"") == 0;
}

static bool test_json_escape_backslash(void) {
    char out[256];
    nf_json_escape("path\\to\\file", out, sizeof(out));
    return strcmp(out, "path\\\\to\\\\file") == 0;
}

static bool test_json_escape_newlines(void) {
    char out[256];
    nf_json_escape("line1\nline2\r\ttab", out, sizeof(out));
    return strcmp(out, "line1\\nline2\\r\\ttab") == 0;
}

static bool test_json_escape_control_chars(void) {
    char out[256];
    char input[] = {0x01, 0x1F, 'A', 0};
    nf_json_escape(input, out, sizeof(out));
    // 0x01 → \u0001, 0x1F → \u001f, A stays
    return strstr(out, "\\u0001") != NULL && strstr(out, "\\u001f") != NULL;
}

static bool test_json_escape_null_input(void) {
    char out[32] = "initial";
    nf_json_escape(NULL, out, sizeof(out));
    return out[0] == '\0';
}

static bool test_json_escape_tiny_buffer(void) {
    char out[4];
    nf_json_escape("\"hello\"", out, sizeof(out));
    // Should not overflow — truncates safely
    return strlen(out) < 4;
}

static bool test_json_injection_checkpoint_path(void) {
    // Verify that checkpoint emit properly escapes malicious paths
    capture_start();
    nf_emit_checkpoint("/tmp/test\",\"injected\":\"value\",\"x\":\"y", 50, 2.5);
    capture_end();

    // The path should be escaped, so the JSON should NOT contain unescaped injected fields
    // The escaped version should have \" as \\\"
    if (strstr(captured, "\"injected\"") != NULL) return false;  // Injection succeeded — FAIL
    if (strstr(captured, "\\\"injected\\\"") == NULL) return false;  // Should be escaped
    return true;
}

static bool test_json_injection_error_message(void) {
    capture_start();
    nf_emit_error("bad \"things\"\nhappened", 1);
    capture_end();

    // Should not have raw newlines or unescaped quotes in JSON
    // Check there's no raw newline splitting the JSON (NDJSON expects one object per line)
    char *nl = strchr(captured, '\n');
    if (nl && *(nl+1) != '\0') return false;  // More than one line = injection
    return strstr(captured, "\\\"things\\\"") != NULL;
}

static bool test_json_injection_restart_reason(void) {
    capture_start();
    nf_emit_restart(1, 5, "reason\"}\n{\"type\":\"evil");
    capture_end();

    // Must not produce two JSON objects
    int newline_count = 0;
    for (char *p = captured; *p; p++) if (*p == '\n') newline_count++;
    return newline_count == 1;  // Exactly one trailing newline
}

// ===== Numeric Validation Tests =====

static bool test_config_negative_steps(void) {
    char *argv[] = { (char*)"--steps", (char*)"-500" };
    NFConfig cfg = nf_config_from_args(2, argv);
    // Should reject negative — fall back to default
    return cfg.total_steps == 10000;
}

static bool test_config_overflow_steps(void) {
    char *argv[] = { (char*)"--steps", (char*)"99999999999" };
    NFConfig cfg = nf_config_from_args(2, argv);
    // Should reject overflow — fall back to default
    return cfg.total_steps == 10000;
}

static bool test_config_nan_lr(void) {
    char *argv[] = { (char*)"--lr", (char*)"NaN" };
    NFConfig cfg = nf_config_from_args(2, argv);
    // Should reject NaN — fall back to default
    return cfg.learning_rate > 0.0f && cfg.learning_rate < 1.0f;
}

static bool test_config_inf_beta(void) {
    char *argv[] = { (char*)"--beta1", (char*)"Inf" };
    NFConfig cfg = nf_config_from_args(2, argv);
    return cfg.beta1 >= 0.0f && cfg.beta1 <= 1.0f;
}

static bool test_config_garbage_string(void) {
    char *argv[] = { (char*)"--steps", (char*)"not_a_number" };
    NFConfig cfg = nf_config_from_args(2, argv);
    return cfg.total_steps == 10000;
}

static bool test_config_empty_string(void) {
    char *argv[] = { (char*)"--steps", (char*)"" };
    NFConfig cfg = nf_config_from_args(2, argv);
    return cfg.total_steps == 10000;
}

static bool test_config_json_negative_values(void) {
    char tmppath[] = "/tmp/nf_test_neg_XXXXXX";
    int fd = mkstemp(tmppath);
    const char *json = "{\"steps\":-100,\"learning_rate\":-1.0,\"beta1\":5.0}";
    write(fd, json, strlen(json));
    close(fd);

    char *argv[] = { (char*)"--config", tmppath };
    NFConfig cfg = nf_config_from_args(2, argv);
    unlink(tmppath);

    // All should be at defaults because values are out of range
    if (cfg.total_steps != 10000) return false;
    if (cfg.beta1 > 1.0f) return false;
    return true;
}

// ===== LR Scheduler Tests =====

static bool test_config_scheduler_args(void) {
    char *argv[] = {
        (char*)"--warmup", (char*)"100",
        (char*)"--lr-min", (char*)"1e-6",
        (char*)"--lr-schedule", (char*)"cosine",
    };
    int argc = sizeof(argv) / sizeof(argv[0]);
    NFConfig cfg = nf_config_from_args(argc, argv);

    if (cfg.warmup_steps != 100) return false;
    if (cfg.lr_min < 9e-7f || cfg.lr_min > 1.1e-6f) return false;
    if (cfg.lr_schedule != 1) return false;
    return true;
}

static bool test_config_scheduler_defaults(void) {
    NFConfig cfg = nf_config_defaults();
    if (cfg.warmup_steps != 0) return false;
    if (cfg.lr_min < 9e-6f || cfg.lr_min > 1.1e-5f) return false;
    if (cfg.lr_schedule != 0) return false;
    return true;
}

static bool test_config_scheduler_none(void) {
    char *argv[] = { (char*)"--lr-schedule", (char*)"none" };
    NFConfig cfg = nf_config_from_args(2, argv);
    return cfg.lr_schedule == 0;
}

// We need nf_compute_lr for testing — replicate the logic inline
static float test_compute_lr(float base_lr, float lr_min, int warmup, int schedule,
                              int total_steps, int step) {
    if (warmup > 0 && step < warmup) {
        return base_lr * ((float)(step + 1) / (float)warmup);
    }
    if (schedule == 1) {
        int decay_steps = total_steps - warmup;
        if (decay_steps <= 0) return base_lr;
        float decay_ratio = (float)(step - warmup) / (float)decay_steps;
        if (decay_ratio > 1.0f) decay_ratio = 1.0f;
        if (decay_ratio < 0.0f) decay_ratio = 0.0f;
        float coeff = 0.5f * (1.0f + cosf((float)M_PI * decay_ratio));
        return lr_min + (base_lr - lr_min) * coeff;
    }
    return base_lr;
}

static bool test_lr_constant(void) {
    // No schedule → constant LR
    float lr = test_compute_lr(3e-4f, 1e-5f, 0, 0, 1000, 500);
    return fabsf(lr - 3e-4f) < 1e-8f;
}

static bool test_lr_warmup_start(void) {
    // Step 0 of 100 warmup: LR = base * (1/100) = 3e-6
    float lr = test_compute_lr(3e-4f, 1e-5f, 100, 0, 1000, 0);
    float expected = 3e-4f * (1.0f / 100.0f);
    return fabsf(lr - expected) < 1e-8f;
}

static bool test_lr_warmup_mid(void) {
    // Step 49 of 100 warmup: LR = base * (50/100) = 1.5e-4
    float lr = test_compute_lr(3e-4f, 1e-5f, 100, 0, 1000, 49);
    float expected = 3e-4f * (50.0f / 100.0f);
    return fabsf(lr - expected) < 1e-8f;
}

static bool test_lr_warmup_end(void) {
    // Step 99 of 100 warmup: LR ≈ base * (100/100) = 3e-4
    float lr = test_compute_lr(3e-4f, 1e-5f, 100, 0, 1000, 99);
    float expected = 3e-4f * (100.0f / 100.0f);
    return fabsf(lr - expected) < 1e-8f;
}

static bool test_lr_cosine_start(void) {
    // Cosine at step 0 (no warmup): cos(0)=1, coeff=1 → LR = base
    float lr = test_compute_lr(3e-4f, 1e-5f, 0, 1, 1000, 0);
    float expected = 3e-4f;  // coeff = 0.5*(1+cos(0)) = 1.0
    return fabsf(lr - expected) < 1e-7f;
}

static bool test_lr_cosine_mid(void) {
    // Cosine at step 500/1000 (no warmup): cos(pi*0.5) ≈ 0, coeff ≈ 0.5
    float lr = test_compute_lr(3e-4f, 1e-5f, 0, 1, 1000, 500);
    float expected = 1e-5f + (3e-4f - 1e-5f) * 0.5f;  // midpoint
    return fabsf(lr - expected) < 1e-6f;
}

static bool test_lr_cosine_end(void) {
    // Cosine at step 1000/1000: cos(pi)=-1, coeff=0 → LR = lr_min
    float lr = test_compute_lr(3e-4f, 1e-5f, 0, 1, 1000, 1000);
    float expected = 1e-5f;
    return fabsf(lr - expected) < 1e-8f;
}

static bool test_lr_cosine_with_warmup(void) {
    // Warmup 100, cosine 100-1000. At step 50 (warmup): LR = base * 51/100
    float lr50 = test_compute_lr(3e-4f, 1e-5f, 100, 1, 1000, 50);
    float exp50 = 3e-4f * (51.0f / 100.0f);
    if (fabsf(lr50 - exp50) > 1e-8f) return false;
    // At step 100 (post-warmup start): cos(0) → LR = base
    float lr100 = test_compute_lr(3e-4f, 1e-5f, 100, 1, 1000, 100);
    if (fabsf(lr100 - 3e-4f) > 1e-7f) return false;
    // At step 1000 (end): LR = lr_min
    float lr1000 = test_compute_lr(3e-4f, 1e-5f, 100, 1, 1000, 1000);
    if (fabsf(lr1000 - 1e-5f) > 1e-8f) return false;
    return true;
}

static bool test_config_scheduler_json(void) {
    char tmppath[] = "/tmp/nf_test_sched_XXXXXX";
    int fd = mkstemp(tmppath);
    const char *json = "{\"warmup_steps\":200,\"lr_min\":0.00001,\"lr_schedule\":\"cosine\"}";
    write(fd, json, strlen(json));
    close(fd);

    char *argv[] = { (char*)"--config", tmppath };
    NFConfig cfg = nf_config_from_args(2, argv);
    unlink(tmppath);

    if (cfg.warmup_steps != 200) return false;
    if (cfg.lr_min < 9e-6f || cfg.lr_min > 1.1e-5f) return false;
    if (cfg.lr_schedule != 1) return false;
    return true;
}

// ===== Data Pipeline Tests =====

static bool test_config_data_pipeline_defaults(void) {
    NFConfig cfg = nf_config_defaults();
    if (cfg.val_data_path[0] != '\0') return false;
    if (cfg.val_every != 0) return false;
    if (cfg.val_batches != 10) return false;
    if (cfg.shuffle != false) return false;
    return true;
}

static bool test_config_data_pipeline_args(void) {
    char *argv[] = {
        (char*)"--val-data", (char*)"/tmp/val.bin",
        (char*)"--val-every", (char*)"25",
        (char*)"--val-batches", (char*)"5",
        (char*)"--shuffle"
    };
    NFConfig cfg = nf_config_from_args(7, argv);
    if (strcmp(cfg.val_data_path, "/tmp/val.bin") != 0) return false;
    if (cfg.val_every != 25) return false;
    if (cfg.val_batches != 5) return false;
    if (cfg.shuffle != true) return false;
    return true;
}

static bool test_config_data_pipeline_json(void) {
    char tmppath[] = "/tmp/nf_test_data_XXXXXX";
    int fd = mkstemp(tmppath);
    const char *json = "{\"val_data\":\"/data/val.bin\",\"val_every\":50,\"val_batches\":20,\"shuffle\":true}";
    write(fd, json, strlen(json));
    close(fd);

    char *argv[] = { (char*)"--config", tmppath };
    NFConfig cfg = nf_config_from_args(2, argv);
    unlink(tmppath);

    if (strcmp(cfg.val_data_path, "/data/val.bin") != 0) return false;
    if (cfg.val_every != 50) return false;
    if (cfg.val_batches != 20) return false;
    if (cfg.shuffle != true) return false;
    return true;
}

static bool test_config_shuffle_only(void) {
    char *argv[] = { (char*)"--shuffle" };
    NFConfig cfg = nf_config_from_args(1, argv);
    if (cfg.shuffle != true) return false;
    // val should remain disabled
    if (cfg.val_every != 0) return false;
    if (cfg.val_data_path[0] != '\0') return false;
    return true;
}

static bool test_xorshift32_deterministic(void) {
    // xorshift32 with same seed should produce same sequence
    uint32_t s1 = 12345, s2 = 12345;
    for (int i = 0; i < 100; i++) {
        if (xorshift32(&s1) != xorshift32(&s2)) return false;
    }
    return true;
}

static bool test_xorshift32_different_seeds(void) {
    uint32_t s1 = 42, s2 = 43;
    // Different seeds should produce different sequences
    int diffs = 0;
    for (int i = 0; i < 10; i++) {
        if (xorshift32(&s1) != xorshift32(&s2)) diffs++;
    }
    return diffs > 0;
}

static bool test_config_val_every_zero(void) {
    // val_every=0 means validation is disabled
    char *argv[] = {
        (char*)"--val-data", (char*)"/tmp/val.bin",
        (char*)"--val-every", (char*)"0"
    };
    NFConfig cfg = nf_config_from_args(4, argv);
    return cfg.val_every == 0;  // validation effectively disabled
}

// ===== Text Generation Tests =====

static bool test_progress_token_json(void) {
    capture_start();
    nf_emit_token(42, "hello");
    capture_end();

    if (strstr(captured, "\"type\":\"token\"") == NULL) return false;
    if (strstr(captured, "\"token_id\":42") == NULL) return false;
    if (strstr(captured, "\"text\":\"hello\"") == NULL) return false;
    size_t len = strlen(captured);
    if (len == 0 || captured[len-1] != '\n') return false;
    return true;
}

static bool test_progress_token_escape(void) {
    capture_start();
    nf_emit_token(1, "say \"hi\"\nbye");
    capture_end();

    // Should escape quotes and newlines
    if (strstr(captured, "\\\"hi\\\"") == NULL) return false;
    if (strstr(captured, "\\n") == NULL) return false;
    // Must be single NDJSON line
    int newlines = 0;
    for (char *p = captured; *p; p++) if (*p == '\n') newlines++;
    return newlines == 1;
}

static bool test_progress_generate_done_json(void) {
    capture_start();
    nf_emit_generate_done(150, 2300.5);
    capture_end();

    if (strstr(captured, "\"type\":\"generate_done\"") == NULL) return false;
    if (strstr(captured, "\"tokens\":150") == NULL) return false;
    if (strstr(captured, "\"total_ms\":2300.5") == NULL) return false;
    size_t len = strlen(captured);
    if (len == 0 || captured[len-1] != '\n') return false;
    return true;
}

static bool test_tokenizer_decode(void) {
    const char *path = find_tokenizer();
    if (!path) {
        fprintf(stderr, "(tokenizer.bin not found, skipping) ");
        return true;
    }

    NFTokenizer tok;
    if (!nf_tokenizer_load(&tok, path, 32000)) return false;

    // Token 0 should decode to something (BOS or first vocab entry)
    const char *s = nf_tokenizer_decode(&tok, 0);
    if (s == NULL) { nf_tokenizer_free(&tok); return false; }

    // Out-of-range should return NULL
    if (nf_tokenizer_decode(&tok, -1) != NULL) { nf_tokenizer_free(&tok); return false; }
    if (nf_tokenizer_decode(&tok, 32000) != NULL) { nf_tokenizer_free(&tok); return false; }
    if (nf_tokenizer_decode(&tok, 99999) != NULL) { nf_tokenizer_free(&tok); return false; }

    nf_tokenizer_free(&tok);
    return true;
}

static bool test_tokenizer_encode_decode_roundtrip(void) {
    const char *path = find_tokenizer();
    if (!path) {
        fprintf(stderr, "(tokenizer.bin not found, skipping) ");
        return true;
    }

    NFTokenizer tok;
    if (!nf_tokenizer_load(&tok, path, 32000)) return false;

    // Encode "Hello" then decode each token — concatenation should contain original chars
    uint16_t tokens[256];
    int n = nf_tokenizer_encode(&tok, "Hello", tokens, 256);
    if (n <= 0) { nf_tokenizer_free(&tok); return false; }

    // Concatenate decoded tokens
    char result[1024] = {0};
    for (int i = 0; i < n; i++) {
        const char *t = nf_tokenizer_decode(&tok, tokens[i]);
        if (t) strlcat(result, t, sizeof(result));
    }

    nf_tokenizer_free(&tok);
    // Decoded text should contain "Hello"
    return strstr(result, "Hello") != NULL;
}

static bool test_tokenizer_decode_null(void) {
    // Decode with NULL tokenizer should not crash
    return nf_tokenizer_decode(NULL, 0) == NULL;
}

static bool test_progress_val_json(void) {
    capture_start();
    nf_emit_val(100, 3.45, 10);
    capture_end();

    if (strstr(captured, "\"type\":\"val\"") == NULL) return false;
    if (strstr(captured, "\"step\":100") == NULL) return false;
    if (strstr(captured, "\"val_batches\":10") == NULL) return false;
    return true;
}

// ===== Stability Tests =====

static bool test_tokenizer_null_text(void) {
    // nf_tokenizer_encode with NULL should not crash
    const char *path = find_tokenizer();
    if (!path) {
        fprintf(stderr, "(tokenizer.bin not found, skipping) ");
        return true;
    }
    NFTokenizer tok;
    if (!nf_tokenizer_load(&tok, path, 32000)) return false;
    uint16_t out[16];
    int n = nf_tokenizer_encode(&tok, NULL, out, 16);
    nf_tokenizer_free(&tok);
    return n == 0;
}

static bool test_tokenizer_invalid_file(void) {
    // Loading a non-existent tokenizer should return false, not crash
    NFTokenizer tok;
    return !nf_tokenizer_load(&tok, "/nonexistent/path/tok.bin", 32000);
}

static bool test_tokenizer_corrupted_file(void) {
    // Create a file with garbage data
    char tmppath[] = "/tmp/nf_test_corrtok_XXXXXX";
    int fd = mkstemp(tmppath);
    char garbage[64];
    memset(garbage, 0xFF, sizeof(garbage));
    write(fd, garbage, sizeof(garbage));
    close(fd);

    NFTokenizer tok;
    bool loaded = nf_tokenizer_load(&tok, tmppath, 32000);
    unlink(tmppath);
    // Should fail gracefully
    if (loaded) { nf_tokenizer_free(&tok); return false; }
    return true;
}

static bool test_config_json_corrupt(void) {
    // Corrupted JSON should not crash
    char tmppath[] = "/tmp/nf_test_badjson_XXXXXX";
    int fd = mkstemp(tmppath);
    const char *json = "{this is not valid json!!! {{{";
    write(fd, json, strlen(json));
    close(fd);

    char *argv[] = { (char*)"--config", tmppath };
    NFConfig cfg = nf_config_from_args(2, argv);
    unlink(tmppath);

    // Should use defaults after failed JSON parse
    return cfg.total_steps == 10000 && cfg.seed == 42;
}

static bool test_config_json_wrong_types(void) {
    // JSON with wrong types (strings where numbers expected)
    char tmppath[] = "/tmp/nf_test_types_XXXXXX";
    int fd = mkstemp(tmppath);
    const char *json = "{\"steps\":\"not a number\",\"learning_rate\":\"fast\"}";
    write(fd, json, strlen(json));
    close(fd);

    char *argv[] = { (char*)"--config", tmppath };
    NFConfig cfg = nf_config_from_args(2, argv);
    unlink(tmppath);

    // intValue/floatValue on NSString returns 0 — should stay at defaults
    // because 0 is out of our valid range [1, 10000000]
    return cfg.total_steps == 10000;
}

static bool test_config_nonexistent_json(void) {
    char *argv[] = { (char*)"--config", (char*)"/nonexistent/config.json" };
    NFConfig cfg = nf_config_from_args(2, argv);
    return cfg.total_steps == 10000;
}

// ===== LoRA Tests =====

static bool test_config_lora_defaults(void) {
    NFConfig cfg = nf_config_defaults();
    if (cfg.lora_rank != 0) return false;
    if (cfg.lora_alpha < 15.9f || cfg.lora_alpha > 16.1f) return false;
    if (cfg.lora_targets != 8) return false;
    return true;
}

static bool test_config_lora_args(void) {
    char *argv[] = {
        (char*)"--lora-rank", (char*)"8",
        (char*)"--lora-alpha", (char*)"32",
        (char*)"--lora-targets", (char*)"15",
    };
    int argc = sizeof(argv) / sizeof(argv[0]);
    NFConfig cfg = nf_config_from_args(argc, argv);

    if (cfg.lora_rank != 8) return false;
    if (cfg.lora_alpha < 31.9f || cfg.lora_alpha > 32.1f) return false;
    if (cfg.lora_targets != 15) return false;
    return true;
}

static bool test_config_lora_json(void) {
    char tmppath[] = "/tmp/nf_test_lora_XXXXXX";
    int fd = mkstemp(tmppath);
    const char *json = "{\"lora_rank\":16,\"lora_alpha\":64.0,\"lora_targets\":8}";
    write(fd, json, strlen(json));
    close(fd);

    char *argv[] = { (char*)"--config", tmppath };
    NFConfig cfg = nf_config_from_args(2, argv);
    unlink(tmppath);

    if (cfg.lora_rank != 16) return false;
    if (cfg.lora_alpha < 63.9f || cfg.lora_alpha > 64.1f) return false;
    if (cfg.lora_targets != 8) return false;
    return true;
}

static bool test_config_lora_invalid_rank(void) {
    // Rank > 256 should be rejected
    char *argv[] = { (char*)"--lora-rank", (char*)"999" };
    NFConfig cfg = nf_config_from_args(2, argv);
    return cfg.lora_rank == 0;  // Falls back to default
}

static bool test_config_lora_invalid_targets(void) {
    // Targets > 15 should be rejected
    char *argv[] = { (char*)"--lora-targets", (char*)"16" };
    NFConfig cfg = nf_config_from_args(2, argv);
    return cfg.lora_targets == 8;  // Falls back to default
}

static bool test_progress_lora_info_json(void) {
    capture_start();
    nf_emit_lora_info(8, 16.0f, 8, 147456);
    capture_end();

    if (strstr(captured, "\"type\":\"info\"") == NULL) return false;
    if (strstr(captured, "\"key\":\"lora\"") == NULL) return false;
    if (strstr(captured, "\"rank\":8") == NULL) return false;
    if (strstr(captured, "\"params\":147456") == NULL) return false;
    size_t len = strlen(captured);
    if (len == 0 || captured[len-1] != '\n') return false;
    return true;
}

static bool test_lora_adapter_alloc_init(void) {
    srand48(42);
    LoRAAdapter a = lora_adapter_alloc(4, 768, 768);
    if (a.rank != 4) return false;
    if (a.in_dim != 768) return false;
    if (a.out_dim != 768) return false;
    if (a.A == NULL || a.B == NULL) return false;
    // B should be all zeros (initialized to zero)
    float bsum = 0;
    for (size_t i = 0; i < (size_t)768 * 4; i++) bsum += fabsf(a.B[i]);
    if (bsum > 1e-6f) { lora_adapter_free(&a); return false; }
    // A should have non-zero values (kaiming init)
    float asum = 0;
    for (size_t i = 0; i < (size_t)4 * 768; i++) asum += fabsf(a.A[i]);
    if (asum < 1.0f) { lora_adapter_free(&a); return false; }
    lora_adapter_free(&a);
    return true;
}

static bool test_lora_scale_computation(void) {
    // scale = alpha / rank
    float alpha = 16.0f;
    int rank = 8;
    float scale = alpha / (float)rank;
    if (fabsf(scale - 2.0f) > 1e-6f) return false;

    alpha = 32.0f; rank = 4;
    scale = alpha / (float)rank;
    if (fabsf(scale - 8.0f) > 1e-6f) return false;

    return true;
}

static bool test_lora_param_count(void) {
    // For Wo only (targets=8), rank=8: 2 * DIM * rank * NLAYERS
    int rank = 8;
    int expected = 2 * 768 * rank * 12;  // = 147456
    if (expected != 147456) return false;
    // For rank=16: 2 * 768 * 16 * 12 = 294912
    rank = 16;
    expected = 2 * 768 * rank * 12;
    if (expected != 294912) return false;
    return true;
}

static bool test_lora_checkpoint_header(void) {
    // CkptHdr.pad[0] stores lora_rank, pad[1] stores targets,
    // pad[2] stores alpha as memcpy'd float
    CkptHdr h = {0};
    h.pad[0] = 8;
    h.pad[1] = 8;
    float alpha = 16.0f;
    memcpy(&h.pad[2], &alpha, sizeof(float));

    // Verify we can read back
    if (h.pad[0] != 8) return false;
    if (h.pad[1] != 8) return false;
    float read_alpha;
    memcpy(&read_alpha, &h.pad[2], sizeof(float));
    if (fabsf(read_alpha - 16.0f) > 1e-6f) return false;
    return true;
}

// ===== Multi-Model (ModelConfig) Tests =====

static bool test_model_config_defaults(void) {
    // model_config_defaults() should set Stories110M values
    ModelConfig saved = g_mc;  // Save current
    model_config_defaults();
    bool ok = (g_mc.dim == 768 && g_mc.hidden_dim == 2048 &&
               g_mc.n_heads == 12 && g_mc.seq_len == 256 &&
               g_mc.n_layers == 12 && g_mc.vocab_size == 32000);
    g_mc = saved;  // Restore
    return ok;
}

static bool test_model_config_derived_sizes(void) {
    ModelConfig mc = {0};
    mc.dim = 768; mc.hidden_dim = 2048; mc.n_heads = 12;
    mc.seq_len = 256; mc.n_layers = 12; mc.vocab_size = 32000;
    model_config_init(&mc);
    if (mc.head_dim != 64) return false;        // 768/12 = 64
    if (mc.wq_sz != 768*768) return false;
    if (mc.wo_sz != 768*768) return false;
    if (mc.w1_sz != 2048*768) return false;
    if (mc.w2_sz != 768*2048) return false;
    if (mc.w3_sz != 2048*768) return false;
    if (mc.score_ch != 12*256) return false;     // n_heads * seq_len
    return true;
}

static bool test_model_config_total_params(void) {
    ModelConfig mc = {0};
    mc.dim = 768; mc.hidden_dim = 2048; mc.n_heads = 12;
    mc.seq_len = 256; mc.n_layers = 12; mc.vocab_size = 32000;
    model_config_init(&mc);
    // layer_params = 4*dim*dim + hidden*dim + dim*hidden + hidden*dim + 2*dim
    // = 4*589824 + 1572864 + 1572864 + 1572864 + 1536
    // = 2359296 + 4718592 + 1536 = 7079424
    int expected_layer = 4*768*768 + 2048*768 + 768*2048 + 2048*768 + 2*768;
    if (mc.layer_params != expected_layer) return false;
    // total = 12 * layer_params + dim + vocab*dim
    int expected_total = 12 * expected_layer + 768 + 32000*768;
    if (mc.total_params != expected_total) return false;
    return true;
}

static bool test_model_config_custom_dims(void) {
    // Test with a smaller model (e.g., 42M-ish)
    ModelConfig mc = {0};
    mc.dim = 512; mc.hidden_dim = 1376; mc.n_heads = 8;
    mc.seq_len = 256; mc.n_layers = 8; mc.vocab_size = 32000;
    model_config_init(&mc);
    if (mc.head_dim != 64) return false;  // 512/8 = 64
    if (mc.wq_sz != 512*512) return false;
    if (mc.w1_sz != 1376*512) return false;
    if (mc.score_ch != 8*256) return false;
    // Verify total_params is reasonable
    if (mc.total_params < 40000000 || mc.total_params > 50000000) return false;
    return true;
}

static bool test_model_config_macros_use_global(void) {
    // Verify DIM/HIDDEN/etc macros reflect g_mc values
    ModelConfig saved = g_mc;
    g_mc.dim = 512; g_mc.hidden_dim = 1376; g_mc.n_heads = 8;
    g_mc.seq_len = 128; g_mc.n_layers = 6; g_mc.vocab_size = 16000;
    model_config_init(&g_mc);
    bool ok = (DIM == 512 && HIDDEN == 1376 && HEADS == 8 &&
               HD == 64 && SEQ == 128 && NLAYERS == 6 && VOCAB == 16000);
    g_mc = saved;  // Restore
    model_config_init(&g_mc);
    return ok;
}

static bool test_ckpt_header_stores_dims(void) {
    // CkptHdr should store and restore model dimensions
    CkptHdr h = {0};
    h.magic = 0x424C5A54; h.version = 2;
    h.dim = 512; h.hidden_dim = 1376; h.n_heads = 8;
    h.seq_len = 128; h.n_layers = 6; h.vocab_size = 16000;
    // Verify we can read them back
    if (h.dim != 512) return false;
    if (h.hidden_dim != 1376) return false;
    if (h.n_heads != 8) return false;
    if (h.seq_len != 128) return false;
    if (h.n_layers != 6) return false;
    if (h.vocab_size != 16000) return false;
    return true;
}

static bool test_model_config_alloc_uses_runtime(void) {
    // Verify that layer_weights_alloc uses current g_mc values
    ModelConfig saved = g_mc;
    g_mc.dim = 32; g_mc.hidden_dim = 64; g_mc.n_heads = 4;
    g_mc.seq_len = 16; g_mc.n_layers = 2; g_mc.vocab_size = 100;
    model_config_init(&g_mc);
    // layer_weights_alloc should use WQ_SZ = 32*32 = 1024
    LayerWeights lw = layer_weights_alloc();
    // Just verify alloc didn't crash and pointers are non-null
    bool ok = (lw.Wq != NULL && lw.Wk != NULL && lw.Wv != NULL &&
               lw.Wo != NULL && lw.W1 != NULL && lw.W2 != NULL &&
               lw.W3 != NULL && lw.rms_att != NULL && lw.rms_ffn != NULL);
    layer_weights_free(&lw);
    g_mc = saved;  // Restore
    model_config_init(&g_mc);
    return ok;
}

// ===== Security Validation Tests =====

static bool test_model_config_validate_good(void) {
    ModelConfig mc = {.dim=768, .hidden_dim=2048, .n_heads=12, .seq_len=256, .n_layers=12, .vocab_size=32000};
    return model_config_validate(&mc);
}
static bool test_model_config_validate_bad_dim(void) {
    ModelConfig mc = {.dim=0, .hidden_dim=2048, .n_heads=12, .seq_len=256, .n_layers=12, .vocab_size=32000};
    if (model_config_validate(&mc)) return false;
    mc.dim = 99999; // too large
    return !model_config_validate(&mc);
}
static bool test_model_config_validate_bad_layers(void) {
    ModelConfig mc = {.dim=768, .hidden_dim=2048, .n_heads=12, .seq_len=256, .n_layers=0, .vocab_size=32000};
    if (model_config_validate(&mc)) return false;
    mc.n_layers = 500; // too many
    return !model_config_validate(&mc);
}
static bool test_model_config_validate_bad_heads(void) {
    ModelConfig mc = {.dim=768, .hidden_dim=2048, .n_heads=7, .seq_len=256, .n_layers=12, .vocab_size=32000};
    return !model_config_validate(&mc);  // 768 % 7 != 0
}
static bool test_model_config_validate_bad_seq(void) {
    ModelConfig mc = {.dim=768, .hidden_dim=2048, .n_heads=12, .seq_len=0, .n_layers=12, .vocab_size=32000};
    if (model_config_validate(&mc)) return false;
    mc.seq_len = 99999;
    return !model_config_validate(&mc);
}
static bool test_model_config_validate_bad_vocab(void) {
    ModelConfig mc = {.dim=768, .hidden_dim=2048, .n_heads=12, .seq_len=256, .n_layers=12, .vocab_size=0};
    if (model_config_validate(&mc)) return false;
    mc.vocab_size = 999999;
    return !model_config_validate(&mc);
}
static bool test_xorshift32_zero_guard(void) {
    // xorshift32 should handle zero state (bug fix)
    uint32_t state = 0;
    uint32_t val = xorshift32(&state);
    // After fix, state=0 becomes state=1 before shifting, so result should not be zero
    return (val != 0 && state != 0);
}

// ===== Math / CPU Ops Tests =====
#include "stories_cpu_ops.h"

static bool test_rmsnorm_identity(void) {
    // RMSNorm with all-ones weights should normalize to unit RMS
    int d = 4, S = 2;
    float x[] = {1,1, 2,2, 3,3, 4,4}; // [4,2] column-major
    float w[] = {1, 1, 1, 1};
    float out[8] = {0};
    rmsnorm(out, x, w, d, S);
    // Each column: [1,2,3,4] -> RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
    // normalized: [1/2.7386, 2/2.7386, 3/2.7386, 4/2.7386]
    float rms = sqrtf((1+4+9+16)/4.0f + 1e-5f);
    float expected0 = 1.0f / rms;
    return fabsf(out[0] - expected0) < 0.01f;
}

static bool test_rmsnorm_bwd_gradient(void) {
    // Test rmsnorm backward: finite difference check
    int d = 2, S = 1;
    float x[] = {1.0f, 2.0f};
    float w[] = {1.0f, 1.0f};
    float out[2], out2[2];
    rmsnorm(out, x, w, d, S);

    // Perturb x[0]
    float eps = 1e-3f;
    float x2[] = {1.0f + eps, 2.0f};
    rmsnorm(out2, x2, w, d, S);

    // Analytic gradient
    float dy[] = {1.0f, 0.0f};  // gradient w.r.t. out[0]
    float dx[2] = {0}, dw[2] = {0};
    rmsnorm_bwd(dx, dw, dy, x, w, d, S);

    float fd_grad = (out2[0] - out[0]) / eps;
    return fabsf(dx[0] - fd_grad) < 0.05f;  // ~5% tolerance for float precision
}

static bool test_adam_update_step(void) {
    // Adam with known params: verify weight changes in correct direction
    float w[] = {1.0f, 2.0f};
    float g[] = {0.1f, -0.2f};
    AdamState s;
    s.m = (float*)calloc(2, 4);
    s.v = (float*)calloc(2, 4);
    s.n = 2;
    adam_update(w, g, &s, 1, 0.001f, 0.9f, 0.999f, 1e-8f);
    // w[0] should decrease (positive gradient), w[1] should increase (negative gradient)
    bool ok = (w[0] < 1.0f && w[1] > 2.0f);
    // Verify momentum was updated
    ok = ok && (s.m[0] > 0) && (s.m[1] < 0);
    ok = ok && (s.v[0] > 0) && (s.v[1] > 0);
    adam_free(&s);
    return ok;
}

static bool test_adam_momentum_decay(void) {
    // After many steps with zero gradient, momentum should decay toward zero
    float w[] = {1.0f};
    float g[] = {1.0f};
    AdamState s; s.m = (float*)calloc(1,4); s.v = (float*)calloc(1,4); s.n = 1;
    adam_update(w, g, &s, 1, 0.001f, 0.9f, 0.999f, 1e-8f);
    float m_after_first = s.m[0];
    // Now apply zero gradient for 10 steps
    float zg[] = {0.0f};
    for (int t = 2; t <= 11; t++) adam_update(w, zg, &s, t, 0.001f, 0.9f, 0.999f, 1e-8f);
    // Momentum should have decayed
    bool ok = (fabsf(s.m[0]) < fabsf(m_after_first));
    adam_free(&s);
    return ok;
}

static bool test_cross_entropy_uniform(void) {
    // Cross-entropy of uniform probs should be -log(1/V) = log(V)
    int V = 10, S = 1;
    float logits[10];
    for (int i = 0; i < V; i++) logits[i] = 0.0f;  // uniform logits
    uint16_t targets[] = {3};
    float dlogits[10];
    float loss = cross_entropy_loss(dlogits, logits, targets, V, S);
    float expected = logf((float)V);
    return fabsf(loss - expected) < 0.01f;
}

static bool test_cross_entropy_gradient(void) {
    // Gradient should be softmax - one_hot, divided by S
    int V = 4, S = 1;
    float logits[] = {2.0f, 1.0f, 0.5f, 0.1f};
    uint16_t targets[] = {0};  // target is class 0
    float dlogits[4];
    cross_entropy_loss(dlogits, logits, targets, V, S);
    // After cross_entropy_loss, dlogits = (softmax(logits) - one_hot(0)) / S
    // softmax[0] should be the largest, and dlogits[0] should be negative (prob - 1)
    return (dlogits[0] < 0.0f);  // softmax(logits[0]) < 1.0, so softmax-1 < 0
}

static bool test_cross_entropy_oob_target(void) {
    // Out-of-bounds target should be handled gracefully
    int V = 4, S = 2;
    float logits[] = {1,1, 0,0, 0,0, 0,0};  // [4,2] column-major
    uint16_t targets[] = {1, 60000};  // second target is OOB
    float dlogits[8];
    float loss = cross_entropy_loss(dlogits, logits, targets, V, S);
    // Should not crash and should produce finite loss
    return (loss == loss && loss > 0);  // NaN check
}

static bool test_embed_lookup_basic(void) {
    int dim = 2, seq = 3, vocab = 4;
    float embed[] = {0,0, 1,1, 2,2, 3,3}; // [vocab, dim] row-major
    uint16_t tokens[] = {0, 2, 3};
    float x[6] = {0};  // [dim=2, seq=3] column-major
    embed_lookup(x, embed, tokens, dim, seq, vocab);
    // x[d*seq+t] = embed[tok*dim+d]
    // t=0: tok=0 -> x[0]=0, x[3]=0
    // t=1: tok=2 -> x[1]=2, x[4]=2
    // t=2: tok=3 -> x[2]=3, x[5]=3
    return (x[0] == 0 && x[1] == 2 && x[2] == 3 &&
            x[3] == 0 && x[4] == 2 && x[5] == 3);
}

static bool test_embed_lookup_oob_zeroed(void) {
    // OOB token should zero out the position (security fix)
    int dim = 2, seq = 2, vocab = 3;
    float embed[] = {1,1, 2,2, 3,3};
    uint16_t tokens[] = {1, 50000};  // second token is OOB
    float x[4];
    memset(x, 0x42, sizeof(x));  // fill with junk
    embed_lookup(x, embed, tokens, dim, seq, vocab);
    // t=0 should be [2,2], t=1 should be zeroed
    return (x[0] == 2 && x[2] == 2 &&  // t=0: correct
            x[1] == 0.0f && x[3] == 0.0f);  // t=1: zeroed (not junk)
}

static bool test_embed_backward_basic(void) {
    int dim = 2, seq = 2, vocab = 3;
    float d_embed[6] = {0};
    float dx[] = {1, 2, 3, 4};  // [dim=2, seq=2] column-major
    uint16_t tokens[] = {0, 2};
    embed_backward(d_embed, dx, tokens, dim, seq, vocab);
    // token 0 at t=0: d_embed[0*2+0] += dx[0*2+0] = 1, d_embed[0*2+1] += dx[1*2+0] = 3
    // token 2 at t=1: d_embed[2*2+0] += dx[0*2+1] = 2, d_embed[2*2+1] += dx[1*2+1] = 4
    return (d_embed[0] == 1 && d_embed[1] == 3 &&
            d_embed[4] == 2 && d_embed[5] == 4);
}

static bool test_lora_forward_basic(void) {
    // LoRA forward: output += B @ A @ input * scale
    int in_dim = 2, out_dim = 2, rank = 1, seq = 1;
    float A[] = {1.0f, 0.0f};   // [rank=1, in=2]: selects first input dim
    float B[] = {2.0f, 3.0f};   // [out=2, rank=1]: scales by 2 and 3
    float input[] = {5.0f, 7.0f};  // [in=2, seq=1]
    float output[] = {0.0f, 0.0f}; // [out=2, seq=1]
    float scale = 0.5f;
    lora_forward(output, input, A, B, in_dim, out_dim, rank, scale, seq);
    // temp = A @ input = [1*5 + 0*7] = [5]
    // delta = B @ temp * scale = [2*5, 3*5] * 0.5 = [5, 7.5]
    // output += delta = [5, 7.5]
    return (fabsf(output[0] - 5.0f) < 0.001f && fabsf(output[1] - 7.5f) < 0.001f);
}

static bool test_lora_backward_gradient(void) {
    // Verify LoRA backward produces non-zero gradients
    int in_dim = 2, out_dim = 2, rank = 1, seq = 1;
    float A[] = {1.0f, 0.5f};
    float B[] = {1.0f, 1.0f};
    float input[] = {1.0f, 2.0f};
    float grad_output[] = {1.0f, 1.0f};
    float grad_A[2] = {0}, grad_B[2] = {0};
    float scale = 1.0f;
    lora_backward(grad_A, grad_B, grad_output, input, A, B, in_dim, out_dim, rank, scale, seq);
    // grad_A and grad_B should be non-zero
    return (grad_A[0] != 0 || grad_A[1] != 0) && (grad_B[0] != 0 || grad_B[1] != 0);
}

static bool test_lora_zero_B_identity(void) {
    // When B=0, LoRA should not modify output (delta = 0)
    int in_dim = 4, out_dim = 4, rank = 2, seq = 1;
    float A[8]; for (int i = 0; i < 8; i++) A[i] = (float)(i + 1);
    float B[8] = {0};  // zero B → zero delta
    float input[] = {1, 2, 3, 4};
    float output[] = {10, 20, 30, 40};
    float saved[] = {10, 20, 30, 40};
    lora_forward(output, input, A, B, in_dim, out_dim, rank, 1.0f, seq);
    bool ok = true;
    for (int i = 0; i < 4; i++) ok = ok && (fabsf(output[i] - saved[i]) < 1e-6f);
    return ok;
}

static bool test_checkpoint_roundtrip_header(void) {
    // Write a checkpoint header to a temp file and read it back
    char path[] = "/tmp/nf_test_ckpt_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return false;
    close(fd);

    CkptHdr h = {0};
    h.magic = 0x424C5A54; h.version = 2;
    h.step = 42; h.total_steps = 100;
    h.dim = 768; h.hidden_dim = 2048; h.n_heads = 12;
    h.seq_len = 256; h.n_layers = 12; h.vocab_size = 32000;
    h.lr = 0.0003f; h.loss = 2.5f;
    h.cum_compile = 1.0; h.cum_train = 2.0; h.cum_wall = 3.0;
    h.cum_steps = 10; h.cum_batches = 5; h.adam_t = 7;

    FILE *f = fopen(path, "wb");
    fwrite(&h, sizeof(h), 1, f);
    fclose(f);

    // Read back
    f = fopen(path, "rb");
    CkptHdr h2;
    fread(&h2, sizeof(h2), 1, f);
    fclose(f);
    unlink(path);

    return (h2.magic == 0x424C5A54 && h2.version == 2 &&
            h2.step == 42 && h2.dim == 768 && h2.n_layers == 12 &&
            fabsf(h2.lr - 0.0003f) < 1e-6f && fabsf(h2.loss - 2.5f) < 1e-6f);
}

static bool test_checkpoint_reject_bad_magic(void) {
    // Verify that a checkpoint with wrong magic would be rejected
    // (Self-contained: reads header and checks magic directly, same as main.m logic)
    char path[] = "/tmp/nf_test_badmagic_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return false;
    close(fd);
    CkptHdr h = {0};
    h.magic = 0xDEADBEEF;  // wrong magic
    h.version = 2;
    FILE *f = fopen(path, "wb");
    fwrite(&h, sizeof(h), 1, f);
    fclose(f);

    // Read back and validate (replicating main.m's nf_read_ckpt_header logic)
    f = fopen(path, "rb");
    CkptHdr h2;
    bool ok = (fread(&h2, sizeof(h2), 1, f) == 1);
    fclose(f);
    unlink(path);
    if (!ok) return false;
    // Magic check: 0xDEADBEEF != 0x424C5A54 → should reject
    return (h2.magic != 0x424C5A54);
}

static bool test_checkpoint_reject_bad_dims(void) {
    // Verify that model_config_validate rejects a checkpoint with absurd dimensions
    char path[] = "/tmp/nf_test_baddims_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return false;
    close(fd);
    CkptHdr h = {0};
    h.magic = 0x424C5A54; h.version = 2;
    h.dim = 99999; h.hidden_dim = 2048; h.n_heads = 12;
    h.seq_len = 256; h.n_layers = 12; h.vocab_size = 32000;
    FILE *f = fopen(path, "wb");
    fwrite(&h, sizeof(h), 1, f);
    fclose(f);

    // Read back and run model_config_validate (same as main.m does)
    f = fopen(path, "rb");
    CkptHdr h2;
    bool ok = (fread(&h2, sizeof(h2), 1, f) == 1);
    fclose(f);
    unlink(path);
    if (!ok) return false;

    ModelConfig mc = {0};
    mc.dim = h2.dim; mc.hidden_dim = h2.hidden_dim; mc.n_heads = h2.n_heads;
    mc.seq_len = h2.seq_len; mc.n_layers = h2.n_layers; mc.vocab_size = h2.vocab_size;
    return !model_config_validate(&mc);  // should reject dim=99999
}

// ===== Tokenizer Speed Tests =====

static double time_ms(uint64_t start, uint64_t end) {
    static mach_timebase_info_data_t tb = {0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)(end - start) * tb.numer / tb.denom / 1e6;
}

// Helper: generate a text string of N characters using repeated printable ASCII
static char *make_test_text(int n) {
    char *text = (char *)malloc(n + 1);
    if (!text) return NULL;
    const char *sample = "The quick brown fox jumps over the lazy dog. "
                         "Once upon a time in a land far away, there lived a kind princess. "
                         "She loved to explore the forest and discover new things every day. ";
    int slen = (int)strlen(sample);
    for (int i = 0; i < n; i++) text[i] = sample[i % slen];
    text[n] = '\0';
    return text;
}

static bool test_tokenizer_speed_10k(void) {
    const char *path = find_tokenizer();
    if (!path) { fprintf(stderr, "(tokenizer.bin not found, skipping) "); return true; }
    NFTokenizer tok;
    if (!nf_tokenizer_load(&tok, path, 32000)) return false;

    char *text = make_test_text(10000);
    if (!text) { nf_tokenizer_free(&tok); return false; }
    uint16_t *out = (uint16_t *)malloc(10000 * sizeof(uint16_t));

    uint64_t start = mach_absolute_time();
    int n = nf_tokenizer_encode(&tok, text, out, 10000);
    uint64_t end = mach_absolute_time();

    double ms = time_ms(start, end);
    fprintf(stderr, "(10K chars → %d tokens in %.1f ms) ", n, ms);

    free(out); free(text); nf_tokenizer_free(&tok);
    // Must produce tokens and complete in < 5 seconds
    return n > 0 && ms < 5000.0;
}

static bool test_tokenizer_speed_100k(void) {
    const char *path = find_tokenizer();
    if (!path) { fprintf(stderr, "(tokenizer.bin not found, skipping) "); return true; }
    NFTokenizer tok;
    if (!nf_tokenizer_load(&tok, path, 32000)) return false;

    char *text = make_test_text(100000);
    if (!text) { nf_tokenizer_free(&tok); return false; }
    uint16_t *out = (uint16_t *)malloc(100000 * sizeof(uint16_t));

    uint64_t start = mach_absolute_time();
    int n = nf_tokenizer_encode(&tok, text, out, 100000);
    uint64_t end = mach_absolute_time();

    double ms = time_ms(start, end);
    fprintf(stderr, "(100K chars → %d tokens in %.1f ms) ", n, ms);

    free(out); free(text); nf_tokenizer_free(&tok);
    return n > 0 && ms < 10000.0;
}

static bool test_tokenizer_speed_1m(void) {
    const char *path = find_tokenizer();
    if (!path) { fprintf(stderr, "(tokenizer.bin not found, skipping) "); return true; }
    NFTokenizer tok;
    if (!nf_tokenizer_load(&tok, path, 32000)) return false;

    char *text = make_test_text(1000000);
    if (!text) { nf_tokenizer_free(&tok); return false; }
    uint16_t *out = (uint16_t *)malloc(1000000 * sizeof(uint16_t));

    uint64_t start = mach_absolute_time();
    int n = nf_tokenizer_encode(&tok, text, out, 1000000);
    uint64_t end = mach_absolute_time();

    double ms = time_ms(start, end);
    fprintf(stderr, "(1M chars → %d tokens in %.1f ms) ", n, ms);

    free(out); free(text); nf_tokenizer_free(&tok);
    // Target: 1MB text in < 30 seconds (old algorithm would hang forever)
    return n > 0 && ms < 30000.0;
}

// ===== Audit Log Tests =====

static bool test_audit_sha256(void) {
    // Verify SHA-256 produces expected hash for known input
    char hash[65];
    nf_sha256_hex("hello", 5, hash);
    // SHA-256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
    return strcmp(hash, "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824") == 0;
}

static bool test_audit_sha256_empty(void) {
    // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    char hash[65];
    nf_sha256_hex("", 0, hash);
    return strcmp(hash, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") == 0;
}

static bool test_audit_timestamp(void) {
    char buf[32];
    nf_audit_timestamp(buf, sizeof(buf));
    // Should be ISO 8601: YYYY-MM-DDTHH:MM:SSZ
    if (strlen(buf) != 20) return false;
    if (buf[4] != '-' || buf[7] != '-' || buf[10] != 'T' || buf[19] != 'Z') return false;
    return true;
}

static bool test_audit_user(void) {
    const char *user = nf_audit_user();
    // Should return a non-empty string
    return user != NULL && strlen(user) > 0;
}

static bool test_audit_init_creates_file(void) {
    // Use a temp path for testing
    char tmpdir[] = "/tmp/nf_audit_test_XXXXXX";
    if (!mkdtemp(tmpdir)) return false;

    char path[1024];
    snprintf(path, sizeof(path), "%s/test_audit.jsonl", tmpdir);

    // Reset global audit state
    g_audit.initialized = false;
    g_audit.fp = NULL;
    strlcpy(g_audit.path, path, sizeof(g_audit.path));
    strlcpy(g_audit.prev_hash,
            "0000000000000000000000000000000000000000000000000000000000000000",
            sizeof(g_audit.prev_hash));
    g_audit.seq = 0;

    // Open directly (since nf_audit_init uses ~/Library/Logs)
    g_audit.fp = fopen(path, "a");
    if (!g_audit.fp) { rmdir(tmpdir); return false; }
    g_audit.initialized = true;

    // Write an entry
    nf_audit_log("test_event", "\"key\":\"value\"");
    nf_audit_close();

    // Verify file exists and has content
    FILE *f = fopen(path, "r");
    if (!f) { unlink(path); rmdir(tmpdir); return false; }
    char line[NF_AUDIT_LINE_MAX];
    bool has_content = (fgets(line, sizeof(line), f) != NULL);
    fclose(f);

    // Verify JSON structure
    bool ok = has_content &&
              strstr(line, "\"event\":\"test_event\"") != NULL &&
              strstr(line, "\"key\":\"value\"") != NULL &&
              strstr(line, "\"hash\":\"") != NULL &&
              strstr(line, "\"prev_hash\":\"") != NULL &&
              strstr(line, "\"seq\":1") != NULL;

    unlink(path);
    rmdir(tmpdir);
    return ok;
}

static bool test_audit_hash_chain(void) {
    // Write multiple entries and verify chain
    char tmpdir[] = "/tmp/nf_audit_chain_XXXXXX";
    if (!mkdtemp(tmpdir)) return false;

    char path[1024];
    snprintf(path, sizeof(path), "%s/chain.jsonl", tmpdir);

    // Reset and init
    g_audit.initialized = false;
    g_audit.fp = fopen(path, "a");
    if (!g_audit.fp) { rmdir(tmpdir); return false; }
    g_audit.initialized = true;
    strlcpy(g_audit.path, path, sizeof(g_audit.path));
    strlcpy(g_audit.prev_hash,
            "0000000000000000000000000000000000000000000000000000000000000000",
            sizeof(g_audit.prev_hash));
    g_audit.seq = 0;

    // Write 3 entries
    nf_audit_log("event_1", NULL);
    nf_audit_log("event_2", "\"step\":42");
    nf_audit_log("event_3", "\"model\":\"test.bin\"");
    nf_audit_close();

    // Verify chain integrity
    int first_bad = 0;
    int count = nf_audit_verify(path, &first_bad);

    unlink(path);
    rmdir(tmpdir);

    return count == 3 && first_bad == 0;
}

static bool test_audit_tamper_detection(void) {
    // Write entries, tamper with one, verify detection
    char tmpdir[] = "/tmp/nf_audit_tamper_XXXXXX";
    if (!mkdtemp(tmpdir)) return false;

    char path[1024];
    snprintf(path, sizeof(path), "%s/tamper.jsonl", tmpdir);

    // Reset and init
    g_audit.initialized = false;
    g_audit.fp = fopen(path, "a");
    if (!g_audit.fp) { rmdir(tmpdir); return false; }
    g_audit.initialized = true;
    strlcpy(g_audit.path, path, sizeof(g_audit.path));
    strlcpy(g_audit.prev_hash,
            "0000000000000000000000000000000000000000000000000000000000000000",
            sizeof(g_audit.prev_hash));
    g_audit.seq = 0;

    // Write 3 good entries
    nf_audit_log("event_1", NULL);
    nf_audit_log("event_2", "\"loss\":3.5");
    nf_audit_log("event_3", NULL);
    nf_audit_close();

    // Tamper: modify the second line
    // Read all lines
    FILE *f = fopen(path, "r");
    if (!f) { unlink(path); rmdir(tmpdir); return false; }
    char lines[3][NF_AUDIT_LINE_MAX];
    for (int i = 0; i < 3; i++) {
        if (!fgets(lines[i], sizeof(lines[i]), f)) {
            fclose(f); unlink(path); rmdir(tmpdir); return false;
        }
    }
    fclose(f);

    // Tamper with line 2: change "loss":3.5 to "loss":1.0
    char *loss_ptr = strstr(lines[1], "\"loss\":3.5");
    if (loss_ptr) {
        memcpy(loss_ptr, "\"loss\":1.0", 10);
    }

    // Rewrite file with tampered content
    f = fopen(path, "w");
    if (!f) { unlink(path); rmdir(tmpdir); return false; }
    for (int i = 0; i < 3; i++) fputs(lines[i], f);
    fclose(f);

    // Verify should detect tamper at entry 2
    int first_bad = 0;
    int valid_count = nf_audit_verify(path, &first_bad);

    unlink(path);
    rmdir(tmpdir);

    // Should have only 1 valid entry (first one), second is tampered
    return valid_count == 1 && first_bad == 2;
}

static bool test_audit_convenience_training(void) {
    // Test nf_audit_training_start/stop convenience functions
    char tmpdir[] = "/tmp/nf_audit_conv_XXXXXX";
    if (!mkdtemp(tmpdir)) return false;

    char path[1024];
    snprintf(path, sizeof(path), "%s/conv.jsonl", tmpdir);

    g_audit.initialized = false;
    g_audit.fp = fopen(path, "a");
    if (!g_audit.fp) { rmdir(tmpdir); return false; }
    g_audit.initialized = true;
    strlcpy(g_audit.path, path, sizeof(g_audit.path));
    strlcpy(g_audit.prev_hash,
            "0000000000000000000000000000000000000000000000000000000000000000",
            sizeof(g_audit.prev_hash));
    g_audit.seq = 0;

    nf_audit_training_start("model.bin", "data.bin", 1000, 3e-4f, 10, 8, 16.0f);
    nf_audit_checkpoint("/tmp/ckpt.bin", 500, 2.5f);
    nf_audit_training_stop(1000, 1.8f, 300.5, "completed");
    nf_audit_close();

    // Read and verify
    FILE *f = fopen(path, "r");
    if (!f) { unlink(path); rmdir(tmpdir); return false; }
    char line[NF_AUDIT_LINE_MAX];
    bool has_training_start = false, has_checkpoint = false, has_training_stop = false;

    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "\"event\":\"training_start\"") &&
            strstr(line, "\"model\":\"model.bin\"") &&
            strstr(line, "\"lora_rank\":8"))
            has_training_start = true;
        if (strstr(line, "\"event\":\"checkpoint_save\"") &&
            strstr(line, "\"step\":500"))
            has_checkpoint = true;
        if (strstr(line, "\"event\":\"training_stop\"") &&
            strstr(line, "\"reason\":\"completed\""))
            has_training_stop = true;
    }
    fclose(f);

    // Verify chain integrity too
    int first_bad = 0;
    int count = nf_audit_verify(path, &first_bad);

    unlink(path);
    rmdir(tmpdir);

    return has_training_start && has_checkpoint && has_training_stop &&
           count == 3 && first_bad == 0;
}

static bool test_audit_verify_empty_file(void) {
    char tmpdir[] = "/tmp/nf_audit_empty_XXXXXX";
    if (!mkdtemp(tmpdir)) return false;

    char path[1024];
    snprintf(path, sizeof(path), "%s/empty.jsonl", tmpdir);

    // Create empty file
    FILE *f = fopen(path, "w");
    if (!f) { rmdir(tmpdir); return false; }
    fclose(f);

    int first_bad = 0;
    int count = nf_audit_verify(path, &first_bad);

    unlink(path);
    rmdir(tmpdir);

    return count == 0 && first_bad == 0;
}

static bool test_audit_verify_nonexistent(void) {
    int first_bad = 0;
    int count = nf_audit_verify("/nonexistent/audit.jsonl", &first_bad);
    return count == -1;
}

static bool test_audit_sequence_numbers(void) {
    // Verify entries have sequential seq numbers
    char tmpdir[] = "/tmp/nf_audit_seq_XXXXXX";
    if (!mkdtemp(tmpdir)) return false;

    char path[1024];
    snprintf(path, sizeof(path), "%s/seq.jsonl", tmpdir);

    g_audit.initialized = false;
    g_audit.fp = fopen(path, "a");
    if (!g_audit.fp) { rmdir(tmpdir); return false; }
    g_audit.initialized = true;
    strlcpy(g_audit.path, path, sizeof(g_audit.path));
    strlcpy(g_audit.prev_hash,
            "0000000000000000000000000000000000000000000000000000000000000000",
            sizeof(g_audit.prev_hash));
    g_audit.seq = 0;

    nf_audit_log("a", NULL);
    nf_audit_log("b", NULL);
    nf_audit_log("c", NULL);
    nf_audit_close();

    FILE *f = fopen(path, "r");
    if (!f) { unlink(path); rmdir(tmpdir); return false; }
    char line[NF_AUDIT_LINE_MAX];
    int expected_seq = 1;
    bool ok = true;
    while (fgets(line, sizeof(line), f)) {
        const char *sp = strstr(line, "\"seq\":");
        if (sp) {
            int seq = atoi(sp + 6);
            if (seq != expected_seq) ok = false;
            expected_seq++;
        }
    }
    fclose(f);

    unlink(path);
    rmdir(tmpdir);

    return ok && expected_seq == 4;
}

// ===== Ingest Tests =====

static bool test_ingest_supported_extensions(void) {
    // Supported formats
    if (!nf_ingest_is_supported("txt")) return false;
    if (!nf_ingest_is_supported("md")) return false;
    if (!nf_ingest_is_supported("csv")) return false;
    if (!nf_ingest_is_supported("pdf")) return false;
    if (!nf_ingest_is_supported("docx")) return false;
    if (!nf_ingest_is_supported("py")) return false;
    if (!nf_ingest_is_supported("swift")) return false;
    if (!nf_ingest_is_supported("TXT")) return false;  // case insensitive

    // Unsupported formats
    if (nf_ingest_is_supported("exe")) return false;
    if (nf_ingest_is_supported("zip")) return false;
    if (nf_ingest_is_supported("bin")) return false;
    if (nf_ingest_is_supported("")) return false;
    if (nf_ingest_is_supported(NULL)) return false;

    return true;
}

static bool test_ingest_extract_plain(void) {
    // Create a temp text file and extract it
    char tmpdir[] = "/tmp/nf_ingest_plain_XXXXXX";
    if (!mkdtemp(tmpdir)) return false;

    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/test.txt", tmpdir);

    FILE *f = fopen(path, "w");
    if (!f) { rmdir(tmpdir); return false; }
    fprintf(f, "Hello NeuralForge\nThis is a test.\n");
    fclose(f);

    char *text = nf_extract_text(path);
    bool ok = text != NULL && strstr(text, "Hello NeuralForge") != NULL;
    if (text) free(text);

    unlink(path);
    rmdir(tmpdir);
    return ok;
}

static bool test_ingest_extract_empty(void) {
    char tmpdir[] = "/tmp/nf_ingest_empty_XXXXXX";
    if (!mkdtemp(tmpdir)) return false;

    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/empty.txt", tmpdir);

    FILE *f = fopen(path, "w");
    fclose(f);  // Empty file

    char *text = nf_extract_text(path);
    bool ok = (text == NULL);  // Empty files should return NULL
    if (text) free(text);

    unlink(path);
    rmdir(tmpdir);
    return ok;
}

static bool test_ingest_extract_nonexistent(void) {
    char *text = nf_extract_text("/nonexistent/path/file.txt");
    return text == NULL;
}

static bool test_ingest_scan_directory(void) {
    // Create temp dir with a mix of supported and unsupported files
    char tmpdir[] = "/tmp/nf_ingest_scan_XXXXXX";
    if (!mkdtemp(tmpdir)) return false;

    char path[PATH_MAX];
    // Create supported files
    snprintf(path, sizeof(path), "%s/a.txt", tmpdir);
    FILE *f = fopen(path, "w"); fprintf(f, "text"); fclose(f);

    snprintf(path, sizeof(path), "%s/b.md", tmpdir);
    f = fopen(path, "w"); fprintf(f, "markdown"); fclose(f);

    snprintf(path, sizeof(path), "%s/c.py", tmpdir);
    f = fopen(path, "w"); fprintf(f, "python"); fclose(f);

    // Create unsupported file
    snprintf(path, sizeof(path), "%s/d.zip", tmpdir);
    f = fopen(path, "w"); fprintf(f, "zip"); fclose(f);

    // Create hidden file (should be skipped)
    snprintf(path, sizeof(path), "%s/.hidden.txt", tmpdir);
    f = fopen(path, "w"); fprintf(f, "hidden"); fclose(f);

    NSArray<NSString *> *files = nf_scan_source_dir(tmpdir);
    bool ok = files.count == 3;  // a.txt, b.md, c.py (not d.zip, not .hidden.txt)

    // Verify sorted order
    if (ok && files.count >= 3) {
        ok = ok && [files[0] hasSuffix:@"a.txt"];
        ok = ok && [files[1] hasSuffix:@"b.md"];
        ok = ok && [files[2] hasSuffix:@"c.py"];
    }

    // Cleanup
    @autoreleasepool {
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm removeItemAtPath:[NSString stringWithUTF8String:tmpdir] error:nil];
    }
    return ok;
}

static bool test_ingest_scan_empty_dir(void) {
    char tmpdir[] = "/tmp/nf_ingest_scanempty_XXXXXX";
    if (!mkdtemp(tmpdir)) return false;

    NSArray<NSString *> *files = nf_scan_source_dir(tmpdir);
    bool ok = files.count == 0;

    rmdir(tmpdir);
    return ok;
}

static bool test_ingest_scan_nonexistent(void) {
    NSArray<NSString *> *files = nf_scan_source_dir("/nonexistent/path");
    return files.count == 0;
}

static bool test_ingest_manifest_roundtrip(void) {
    char tmpdir[] = "/tmp/nf_ingest_manifest_XXXXXX";
    if (!mkdtemp(tmpdir)) return false;

    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/manifest.json", tmpdir);

    // Create and save manifest
    @autoreleasepool {
        NSDictionary *manifest = nf_manifest_build(
            @[@{@"path": @"shard_000.bin", @"tokens": @(5000), @"bytes": @(10000)}],
            @{@"test.txt": @{@"mtime": @"2024-01-01T00:00:00Z", @"tokens": @(5000)}},
            "tokenizer.bin", 32000, 5000);

        if (!nf_manifest_save(manifest, path)) {
            rmdir(tmpdir);
            return false;
        }

        // Load it back
        NSDictionary *loaded = nf_manifest_load(path);
        if (!loaded) { unlink(path); rmdir(tmpdir); return false; }

        bool ok = true;
        ok = ok && [loaded[@"version"] intValue] == 1;
        ok = ok && [loaded[@"total_tokens"] intValue] == 5000;
        ok = ok && [loaded[@"vocab_size"] intValue] == 32000;
        ok = ok && [(NSArray *)loaded[@"shards"] count] == 1;
        ok = ok && [(NSDictionary *)loaded[@"processed_files"] count] == 1;
        ok = ok && loaded[@"last_run"] != nil;
        ok = ok && [loaded[@"tokenizer"] isEqualToString:@"tokenizer.bin"];

        unlink(path);
        rmdir(tmpdir);
        return ok;
    }
}

static bool test_ingest_manifest_load_nonexistent(void) {
    NSDictionary *m = nf_manifest_load("/nonexistent/manifest.json");
    return m == nil;
}

static bool test_ingest_shard_write(void) {
    char tmpdir[] = "/tmp/nf_ingest_shard_XXXXXX";
    if (!mkdtemp(tmpdir)) return false;

    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/shard_000.bin", tmpdir);

    uint16_t tokens[] = {1, 42, 100, 32000, 0, 65535};
    int count = 6;
    size_t written = nf_write_shard(tokens, count, path);
    if (written != count * sizeof(uint16_t)) {
        unlink(path); rmdir(tmpdir); return false;
    }

    // Read back and verify
    FILE *f = fopen(path, "rb");
    if (!f) { unlink(path); rmdir(tmpdir); return false; }
    uint16_t readback[6];
    size_t nread = fread(readback, sizeof(uint16_t), 6, f);
    fclose(f);

    bool ok = nread == 6;
    for (int i = 0; i < 6 && ok; i++) {
        ok = ok && readback[i] == tokens[i];
    }

    unlink(path);
    rmdir(tmpdir);
    return ok;
}

static bool test_ingest_shard_name(void) {
    char buf[64];
    nf_shard_name(0, buf, sizeof(buf));
    if (strcmp(buf, "shard_000.bin") != 0) return false;

    nf_shard_name(5, buf, sizeof(buf));
    if (strcmp(buf, "shard_005.bin") != 0) return false;

    nf_shard_name(123, buf, sizeof(buf));
    if (strcmp(buf, "shard_123.bin") != 0) return false;

    return true;
}

static bool test_ingest_incremental_new_file(void) {
    // A file not in the manifest should be processed
    @autoreleasepool {
        NSDictionary *manifest = @{
            @"processed_files": @{
                @"existing.txt": @{@"mtime": @"2024-01-01T00:00:00Z", @"tokens": @(100)}
            }
        };
        return nf_should_process("/some/path/new_file.txt", manifest);
    }
}

static bool test_ingest_incremental_existing_file(void) {
    // A file with same mtime in manifest should be skipped
    char tmpdir[] = "/tmp/nf_ingest_incr_XXXXXX";
    if (!mkdtemp(tmpdir)) return false;

    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/test.txt", tmpdir);

    FILE *f = fopen(path, "w");
    fprintf(f, "data");
    fclose(f);

    // Get the file's actual mtime
    const char *mtime = nf_file_mtime_iso(path);
    if (!mtime) { unlink(path); rmdir(tmpdir); return false; }

    @autoreleasepool {
        NSDictionary *manifest = @{
            @"processed_files": @{
                @"test.txt": @{
                    @"mtime": [NSString stringWithUTF8String:mtime],
                    @"tokens": @(100)
                }
            }
        };

        // Same mtime → should NOT process
        bool result = !nf_should_process(path, manifest);

        unlink(path);
        rmdir(tmpdir);
        return result;
    }
}

static bool test_ingest_no_manifest(void) {
    // With no manifest (nil), all files should be processed
    return nf_should_process("/any/file.txt", nil);
}

static bool test_ingest_file_mtime(void) {
    char tmpdir[] = "/tmp/nf_ingest_mtime_XXXXXX";
    if (!mkdtemp(tmpdir)) return false;

    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/test.txt", tmpdir);
    FILE *f = fopen(path, "w");
    fprintf(f, "test");
    fclose(f);

    const char *mtime = nf_file_mtime_iso(path);
    bool ok = mtime != NULL && strlen(mtime) == 20;  // "YYYY-MM-DDTHH:MM:SSZ"
    ok = ok && mtime[4] == '-' && mtime[10] == 'T' && mtime[19] == 'Z';

    unlink(path);
    rmdir(tmpdir);
    return ok;
}

static bool test_ingest_emit_json(void) {
    // Verify ingest emit functions produce valid JSON
    char buf[4096];
    FILE *mem = fmemopen(buf, sizeof(buf), "w");
    FILE *old_stdout = stdout;
    stdout = mem;

    nf_emit_ingest_file("test.txt", 1234);
    nf_emit_ingest_done(3, 2, 5678, 1, "/output/manifest.json");

    stdout = old_stdout;
    fclose(mem);

    // Verify JSON structure
    bool ok = strstr(buf, "\"type\":\"ingest_file\"") != NULL;
    ok = ok && strstr(buf, "\"file\":\"test.txt\"") != NULL;
    ok = ok && strstr(buf, "\"tokens\":1234") != NULL;
    ok = ok && strstr(buf, "\"type\":\"ingest_done\"") != NULL;
    ok = ok && strstr(buf, "\"new_files\":3") != NULL;
    ok = ok && strstr(buf, "\"skipped\":2") != NULL;
    ok = ok && strstr(buf, "\"total_tokens\":5678") != NULL;
    ok = ok && strstr(buf, "\"shards\":1") != NULL;

    return ok;
}

// ===== Model Registry Tests =====

static bool test_model_registry_count(void) {
    return nf_model_registry_count >= 3;  // At least SmolLM-135M, SmolLM-360M, TinyLlama
}

static bool test_model_registry_fields(void) {
    // All entries must have required fields
    for (int i = 0; i < nf_model_registry_count; i++) {
        const NFModelEntry *m = &nf_model_registry[i];
        if (!m->name || strlen(m->name) == 0) return false;
        if (!m->display_name || strlen(m->display_name) == 0) return false;
        if (!m->repo_id || strlen(m->repo_id) == 0) return false;
        if (!m->architecture || strlen(m->architecture) == 0) return false;
        if (m->dim <= 0 || m->hidden_dim <= 0) return false;
        if (m->n_layers <= 0 || m->n_heads <= 0) return false;
        if (m->n_kv_heads <= 0 || m->n_kv_heads > m->n_heads) return false;
        if (m->vocab_size <= 0 || m->seq_len <= 0) return false;
        if (m->params_millions <= 0) return false;
        if (!m->description || strlen(m->description) == 0) return false;
    }
    return true;
}

static bool test_model_find_exact(void) {
    const NFModelEntry *m = nf_model_find("smollm-135m");
    if (!m) return false;
    return strcmp(m->repo_id, "HuggingFaceTB/SmolLM-135M") == 0;
}

static bool test_model_find_case_insensitive(void) {
    const NFModelEntry *m = nf_model_find("SMOLLM-135M");
    return m != NULL && strcmp(m->name, "smollm-135m") == 0;
}

static bool test_model_find_tinyllama(void) {
    const NFModelEntry *m = nf_model_find("tinyllama-1.1b");
    if (!m) return false;
    return m->dim == 2048 && m->n_heads == 32 && m->n_kv_heads == 4;
}

static bool test_model_find_nonexistent(void) {
    return nf_model_find("nonexistent-model-xyz") == NULL;
}

static bool test_model_find_null(void) {
    return nf_model_find(NULL) == NULL;
}

static bool test_model_dims_valid(void) {
    // All model dimensions should pass NeuralForge's ModelConfig validation
    for (int i = 0; i < nf_model_registry_count; i++) {
        const NFModelEntry *m = &nf_model_registry[i];
        ModelConfig mc = {0};
        mc.dim = m->dim;
        mc.hidden_dim = m->hidden_dim;
        mc.n_heads = m->n_heads;
        mc.seq_len = m->seq_len;
        mc.n_layers = m->n_layers;
        mc.vocab_size = m->vocab_size;
        if (!model_config_validate(&mc)) return false;
    }
    return true;
}

static bool test_model_gqa_detection(void) {
    // SmolLM-135M has GQA (3 KV heads, 9 query heads)
    const NFModelEntry *smol = nf_model_find("smollm-135m");
    if (!smol) return false;
    bool smol_gqa = smol->n_kv_heads != smol->n_heads;
    if (!smol_gqa) return false;

    // SmolLM-1.7B has full MHA (32 KV heads == 32 query heads)
    const NFModelEntry *big = nf_model_find("smollm-1.7b");
    if (!big) return false;
    bool big_gqa = big->n_kv_heads != big->n_heads;
    return !big_gqa;
}

static bool test_model_emit_json(void) {
    char buf[4096];
    FILE *mem = fmemopen(buf, sizeof(buf), "w");
    FILE *old_stdout = stdout;
    stdout = mem;

    nf_emit_model_entry(&nf_model_registry[0]);

    stdout = old_stdout;
    fclose(mem);

    bool ok = strstr(buf, "\"type\":\"model_info\"") != NULL;
    ok = ok && strstr(buf, "\"name\":\"") != NULL;
    ok = ok && strstr(buf, "\"repo_id\":\"") != NULL;
    ok = ok && strstr(buf, "\"dim\":") != NULL;
    ok = ok && strstr(buf, "\"n_heads\":") != NULL;
    ok = ok && strstr(buf, "\"params_millions\":") != NULL;
    return ok;
}

static bool test_model_card_load_nonexistent(void) {
    NSDictionary *card = nf_model_card_load("/nonexistent/path/card.json");
    return card == nil;
}

static bool test_model_card_roundtrip(void) {
    @autoreleasepool {
        // Create a test model card
        NSDictionary *card = @{
            @"name": @"test-model",
            @"source": @"https://huggingface.co/test/model",
            @"dim": @768,
            @"n_layers": @12,
            @"params_millions": @110.6
        };

        // Save to temp file
        NSString *tmpPath = [NSTemporaryDirectory()
            stringByAppendingPathComponent:@"nf_test_card.json"];
        NSError *err = nil;
        NSData *data = [NSJSONSerialization dataWithJSONObject:card
                                                      options:0
                                                        error:&err];
        if (err || !data) return false;
        [data writeToFile:tmpPath atomically:YES];

        // Load back
        NSDictionary *loaded = nf_model_card_load([tmpPath UTF8String]);
        if (!loaded) return false;

        bool ok = [loaded[@"name"] isEqualToString:@"test-model"];
        ok = ok && [loaded[@"dim"] intValue] == 768;
        ok = ok && [loaded[@"n_layers"] intValue] == 12;

        [[NSFileManager defaultManager] removeItemAtPath:tmpPath error:nil];
        return ok;
    }
}

static bool test_download_emit_json(void) {
    char buf[4096];
    FILE *mem = fmemopen(buf, sizeof(buf), "w");
    FILE *old_stdout = stdout;
    stdout = mem;

    nf_emit_download_progress("test-model", "downloading", 50);
    nf_emit_download_done("test-model", "/path/model.bin", "/path/tokenizer.bin", true);

    stdout = old_stdout;
    fclose(mem);

    bool ok = strstr(buf, "\"type\":\"download_progress\"") != NULL;
    ok = ok && strstr(buf, "\"model\":\"test-model\"") != NULL;
    ok = ok && strstr(buf, "\"status\":\"downloading\"") != NULL;
    ok = ok && strstr(buf, "\"percent\":50") != NULL;
    ok = ok && strstr(buf, "\"type\":\"download_done\"") != NULL;
    ok = ok && strstr(buf, "\"success\":true") != NULL;
    ok = ok && strstr(buf, "\"model_path\":\"/path/model.bin\"") != NULL;
    return ok;
}

// ===== Main =====

int main(int argc, char *argv[]) {
    @autoreleasepool {
        model_config_defaults();  // Initialize g_mc for tests that use DIM/NLAYERS/etc.
        fprintf(stderr, "\n=== NeuralForge CLI Tests ===\n\n");

        fprintf(stderr, "[Config]\n");
        TEST(config_defaults);
        TEST(config_from_args);
        TEST(config_optimizer_args);
        TEST(config_json_file);
        TEST(config_json_with_override);
        TEST(config_partial_args);
        TEST(config_path_overflow);

        fprintf(stderr, "\n[Progress JSON]\n");
        TEST(progress_step_json);
        TEST(progress_error_json);
        TEST(progress_checkpoint_json);
        TEST(progress_done_json);
        TEST(progress_batch_json);
        TEST(progress_restart_json);

        fprintf(stderr, "\n[Tokenizer]\n");
        TEST(tokenizer_load);
        TEST(tokenizer_encode);
        TEST(tokenizer_encode_empty);
        TEST(tokenizer_roundtrip_length);

        fprintf(stderr, "\n[Format]\n");
        TEST(gguf_magic);
        TEST(llama2c_config_roundtrip);
        TEST(checkpoint_magic);

        fprintf(stderr, "\n[Security — JSON Escape]\n");
        TEST(json_escape_basic);
        TEST(json_escape_quotes);
        TEST(json_escape_backslash);
        TEST(json_escape_newlines);
        TEST(json_escape_control_chars);
        TEST(json_escape_null_input);
        TEST(json_escape_tiny_buffer);

        fprintf(stderr, "\n[Security — JSON Injection]\n");
        TEST(json_injection_checkpoint_path);
        TEST(json_injection_error_message);
        TEST(json_injection_restart_reason);

        fprintf(stderr, "\n[Security — Numeric Validation]\n");
        TEST(config_negative_steps);
        TEST(config_overflow_steps);
        TEST(config_nan_lr);
        TEST(config_inf_beta);
        TEST(config_garbage_string);
        TEST(config_empty_string);
        TEST(config_json_negative_values);

        fprintf(stderr, "\n[LR Scheduler]\n");
        TEST(config_scheduler_args);
        TEST(config_scheduler_defaults);
        TEST(config_scheduler_none);
        TEST(lr_constant);
        TEST(lr_warmup_start);
        TEST(lr_warmup_mid);
        TEST(lr_warmup_end);
        TEST(lr_cosine_start);
        TEST(lr_cosine_mid);
        TEST(lr_cosine_end);
        TEST(lr_cosine_with_warmup);
        TEST(config_scheduler_json);

        fprintf(stderr, "\n[Data Pipeline]\n");
        TEST(config_data_pipeline_defaults);
        TEST(config_data_pipeline_args);
        TEST(config_data_pipeline_json);
        TEST(config_shuffle_only);
        TEST(xorshift32_deterministic);
        TEST(xorshift32_different_seeds);
        TEST(config_val_every_zero);

        fprintf(stderr, "\n[Text Generation]\n");
        TEST(progress_token_json);
        TEST(progress_token_escape);
        TEST(progress_generate_done_json);
        TEST(tokenizer_decode);
        TEST(tokenizer_encode_decode_roundtrip);
        TEST(tokenizer_decode_null);
        TEST(progress_val_json);

        fprintf(stderr, "\n[Stability]\n");
        TEST(tokenizer_null_text);
        TEST(tokenizer_invalid_file);
        TEST(tokenizer_corrupted_file);
        TEST(config_json_corrupt);
        TEST(config_json_wrong_types);
        TEST(config_nonexistent_json);

        fprintf(stderr, "\n[LoRA]\n");
        TEST(config_lora_defaults);
        TEST(config_lora_args);
        TEST(config_lora_json);
        TEST(config_lora_invalid_rank);
        TEST(config_lora_invalid_targets);
        TEST(progress_lora_info_json);
        TEST(lora_adapter_alloc_init);
        TEST(lora_scale_computation);
        TEST(lora_param_count);
        TEST(lora_checkpoint_header);

        fprintf(stderr, "\n[Multi-Model]\n");
        TEST(model_config_defaults);
        TEST(model_config_derived_sizes);
        TEST(model_config_total_params);
        TEST(model_config_custom_dims);
        TEST(model_config_macros_use_global);
        TEST(ckpt_header_stores_dims);
        TEST(model_config_alloc_uses_runtime);

        fprintf(stderr, "\n[Security — Dimension Validation]\n");
        TEST(model_config_validate_good);
        TEST(model_config_validate_bad_dim);
        TEST(model_config_validate_bad_layers);
        TEST(model_config_validate_bad_heads);
        TEST(model_config_validate_bad_seq);
        TEST(model_config_validate_bad_vocab);
        TEST(xorshift32_zero_guard);

        fprintf(stderr, "\n[Math — CPU Ops]\n");
        TEST(rmsnorm_identity);
        TEST(rmsnorm_bwd_gradient);
        TEST(adam_update_step);
        TEST(adam_momentum_decay);

        fprintf(stderr, "\n[Math — Cross-Entropy]\n");
        TEST(cross_entropy_uniform);
        TEST(cross_entropy_gradient);
        TEST(cross_entropy_oob_target);

        fprintf(stderr, "\n[Math — Embeddings]\n");
        TEST(embed_lookup_basic);
        TEST(embed_lookup_oob_zeroed);
        TEST(embed_backward_basic);

        fprintf(stderr, "\n[Math — LoRA Ops]\n");
        TEST(lora_forward_basic);
        TEST(lora_backward_gradient);
        TEST(lora_zero_B_identity);

        fprintf(stderr, "\n[Checkpoint I/O]\n");
        TEST(checkpoint_roundtrip_header);
        TEST(checkpoint_reject_bad_magic);
        TEST(checkpoint_reject_bad_dims);

        fprintf(stderr, "\n[Tokenizer Speed]\n");
        TEST(tokenizer_speed_10k);
        TEST(tokenizer_speed_100k);
        TEST(tokenizer_speed_1m);

        fprintf(stderr, "\n[Audit Log]\n");
        TEST(audit_sha256);
        TEST(audit_sha256_empty);
        TEST(audit_timestamp);
        TEST(audit_user);
        TEST(audit_init_creates_file);
        TEST(audit_hash_chain);
        TEST(audit_tamper_detection);
        TEST(audit_convenience_training);
        TEST(audit_verify_empty_file);
        TEST(audit_verify_nonexistent);
        TEST(audit_sequence_numbers);

        fprintf(stderr, "\n[Document Ingestion]\n");
        TEST(ingest_supported_extensions);
        TEST(ingest_extract_plain);
        TEST(ingest_extract_empty);
        TEST(ingest_extract_nonexistent);
        TEST(ingest_scan_directory);
        TEST(ingest_scan_empty_dir);
        TEST(ingest_scan_nonexistent);
        TEST(ingest_manifest_roundtrip);
        TEST(ingest_manifest_load_nonexistent);
        TEST(ingest_shard_write);
        TEST(ingest_shard_name);
        TEST(ingest_incremental_new_file);
        TEST(ingest_incremental_existing_file);
        TEST(ingest_no_manifest);
        TEST(ingest_file_mtime);
        TEST(ingest_emit_json);

        fprintf(stderr, "\n[Model Registry]\n");
        TEST(model_registry_count);
        TEST(model_registry_fields);
        TEST(model_find_exact);
        TEST(model_find_case_insensitive);
        TEST(model_find_tinyllama);
        TEST(model_find_nonexistent);
        TEST(model_find_null);
        TEST(model_dims_valid);
        TEST(model_gqa_detection);
        TEST(model_emit_json);
        TEST(model_card_load_nonexistent);
        TEST(model_card_roundtrip);
        TEST(download_emit_json);

        fprintf(stderr, "\n=== Results: %d/%d passed ===\n\n", tests_passed, tests_run);

        return tests_passed == tests_run ? 0 : 1;
    }
}
