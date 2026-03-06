// test_cli.m — Unit tests for NeuralForge CLI components
//
// Build: make test_cli
// Run:   make test

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include "config.h"
#include "progress.h"
#include "tokenizer.h"
#include "stories_config.h"

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

// ===== Main =====

int main(int argc, char *argv[]) {
    @autoreleasepool {
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

        fprintf(stderr, "\n[Stability]\n");
        TEST(tokenizer_null_text);
        TEST(tokenizer_invalid_file);
        TEST(tokenizer_corrupted_file);
        TEST(config_json_corrupt);
        TEST(config_json_wrong_types);
        TEST(config_nonexistent_json);

        fprintf(stderr, "\n=== Results: %d/%d passed ===\n\n", tests_passed, tests_run);

        return tests_passed == tests_run ? 0 : 1;
    }
}
