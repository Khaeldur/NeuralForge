// models.h — Model registry for NeuralForge
// Defines available pre-trained models with HuggingFace metadata,
// architecture details, and download/conversion support.
//
// Usage:
//   neuralforge models                    — list available models
//   neuralforge download --model <name>   — download and convert a model
#pragma once
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#import <Foundation/Foundation.h>

// ===== Model registry entry =====

typedef struct {
    const char *name;          // Short name (e.g., "smollm-135m")
    const char *display_name;  // Display name (e.g., "SmolLM 135M")
    const char *repo_id;       // HuggingFace repo (e.g., "HuggingFaceTB/SmolLM-135M")
    const char *architecture;  // Architecture type (e.g., "LlamaForCausalLM")
    int dim;                   // Embedding dimension
    int hidden_dim;            // FFN hidden dimension
    int n_layers;              // Number of transformer layers
    int n_heads;               // Number of attention heads
    int n_kv_heads;            // Number of KV heads (< n_heads for GQA)
    int vocab_size;            // Vocabulary size
    int seq_len;               // Context length
    double params_millions;    // Parameter count in millions
    const char *description;   // Short description
    bool gated;                // Requires HF token (e.g., Llama 2)
} NFModelEntry;

// ===== Built-in model registry =====

static const NFModelEntry nf_model_registry[] = {
    {
        .name = "smollm-135m",
        .display_name = "SmolLM 135M",
        .repo_id = "HuggingFaceTB/SmolLM-135M",
        .architecture = "LlamaForCausalLM",
        .dim = 576, .hidden_dim = 1536,
        .n_layers = 30, .n_heads = 9, .n_kv_heads = 3,
        .vocab_size = 49152, .seq_len = 2048,
        .params_millions = 134.7,
        .description = "Tiny model, great for testing. Fast to download and convert.",
        .gated = false,
    },
    {
        .name = "smollm-360m",
        .display_name = "SmolLM 360M",
        .repo_id = "HuggingFaceTB/SmolLM-360M",
        .architecture = "LlamaForCausalLM",
        .dim = 960, .hidden_dim = 2560,
        .n_layers = 32, .n_heads = 15, .n_kv_heads = 5,
        .vocab_size = 49152, .seq_len = 2048,
        .params_millions = 362.0,
        .description = "Small model with good quality/speed balance.",
        .gated = false,
    },
    {
        .name = "smollm-1.7b",
        .display_name = "SmolLM 1.7B",
        .repo_id = "HuggingFaceTB/SmolLM-1.7B",
        .architecture = "LlamaForCausalLM",
        .dim = 2048, .hidden_dim = 8192,
        .n_layers = 24, .n_heads = 32, .n_kv_heads = 32,
        .vocab_size = 49152, .seq_len = 2048,
        .params_millions = 1700.0,
        .description = "Capable model, good for fine-tuning on Apple Silicon.",
        .gated = false,
    },
    {
        .name = "tinyllama-1.1b",
        .display_name = "TinyLlama 1.1B",
        .repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        .architecture = "LlamaForCausalLM",
        .dim = 2048, .hidden_dim = 5632,
        .n_layers = 22, .n_heads = 32, .n_kv_heads = 4,
        .vocab_size = 32000, .seq_len = 2048,
        .params_millions = 1100.0,
        .description = "Compact Llama model trained on 3T tokens. Chat-tuned variant.",
        .gated = false,
    },
    {
        .name = "tinyllama-1.1b-base",
        .display_name = "TinyLlama 1.1B Base",
        .repo_id = "TinyLlama/TinyLlama_v1.1",
        .architecture = "LlamaForCausalLM",
        .dim = 2048, .hidden_dim = 5632,
        .n_layers = 22, .n_heads = 32, .n_kv_heads = 4,
        .vocab_size = 32000, .seq_len = 2048,
        .params_millions = 1100.0,
        .description = "TinyLlama base model (not chat-tuned). Better for custom fine-tuning.",
        .gated = false,
    },
};

static const int nf_model_registry_count =
    sizeof(nf_model_registry) / sizeof(nf_model_registry[0]);

// ===== Registry lookup =====

// Find a model by short name. Returns NULL if not found.
static const NFModelEntry *nf_model_find(const char *name) {
    if (!name) return NULL;
    for (int i = 0; i < nf_model_registry_count; i++) {
        if (strcasecmp(name, nf_model_registry[i].name) == 0)
            return &nf_model_registry[i];
    }
    return NULL;
}

// ===== Model list display =====

// Print available models to stderr in a formatted table.
static void nf_models_list(void) {
    fprintf(stderr, "\nAvailable models:\n\n");
    fprintf(stderr, "  %-22s %-8s %-6s %-6s %-6s  %s\n",
            "NAME", "PARAMS", "DIM", "HEADS", "SEQ", "DESCRIPTION");
    fprintf(stderr, "  %-22s %-8s %-6s %-6s %-6s  %s\n",
            "----", "------", "---", "-----", "---", "-----------");

    for (int i = 0; i < nf_model_registry_count; i++) {
        const NFModelEntry *m = &nf_model_registry[i];
        char params[16];
        if (m->params_millions >= 1000)
            snprintf(params, sizeof(params), "%.1fB", m->params_millions / 1000.0);
        else
            snprintf(params, sizeof(params), "%.0fM", m->params_millions);

        fprintf(stderr, "  %-22s %-8s %-6d %-6d %-6d  %s%s\n",
                m->name, params, m->dim, m->n_heads, m->seq_len,
                m->description, m->gated ? " [requires token]" : "");
    }
    fprintf(stderr, "\nDownload: neuralforge download --model <name> --output <dir>\n");
    fprintf(stderr, "Convert: python3 converters/convert_hf.py --model <hf-repo> --output <dir>\n\n");
}

// ===== JSON emission for model info =====

// Emit a model registry entry as JSON (for app consumption).
static void nf_emit_model_entry(const NFModelEntry *m) {
    fprintf(stdout,
        "{\"type\":\"model_info\","
        "\"name\":\"%s\","
        "\"display_name\":\"%s\","
        "\"repo_id\":\"%s\","
        "\"architecture\":\"%s\","
        "\"dim\":%d,\"hidden_dim\":%d,"
        "\"n_layers\":%d,\"n_heads\":%d,\"n_kv_heads\":%d,"
        "\"vocab_size\":%d,\"seq_len\":%d,"
        "\"params_millions\":%.1f,"
        "\"description\":\"%s\","
        "\"gated\":%s}\n",
        m->name, m->display_name, m->repo_id, m->architecture,
        m->dim, m->hidden_dim,
        m->n_layers, m->n_heads, m->n_kv_heads,
        m->vocab_size, m->seq_len,
        m->params_millions,
        m->description,
        m->gated ? "true" : "false");
    fflush(stdout);
}

// Emit all model entries as JSON array.
static void nf_emit_model_list(void) {
    for (int i = 0; i < nf_model_registry_count; i++) {
        nf_emit_model_entry(&nf_model_registry[i]);
    }
}

// ===== Download helpers =====

// Check if Python3 and required packages are available.
static bool nf_check_python_deps(void) {
    @autoreleasepool {
        NSTask *task = [[NSTask alloc] init];
        task.executableURL = [NSURL fileURLWithPath:@"/usr/bin/python3"];
        task.arguments = @[@"-c", @"import safetensors, transformers; print('ok')"];
        task.standardOutput = [NSPipe pipe];
        task.standardError = [NSFileHandle fileHandleWithNullDevice];

        NSError *err = nil;
        [task launchAndReturnError:&err];
        if (err) return false;
        [task waitUntilExit];
        return task.terminationStatus == 0;
    }
}

// Get the path to the convert_hf.py script relative to the CLI binary.
static NSString *nf_converter_script_path(void) {
    @autoreleasepool {
        // Try relative to binary location
        NSString *binPath = [[NSProcessInfo processInfo].arguments[0] stringByDeletingLastPathComponent];
        NSString *scriptPath = [binPath stringByAppendingPathComponent:@"../converters/convert_hf.py"];
        scriptPath = [scriptPath stringByStandardizingPath];
        if ([[NSFileManager defaultManager] fileExistsAtPath:scriptPath])
            return scriptPath;

        // Try from working directory
        scriptPath = @"converters/convert_hf.py";
        if ([[NSFileManager defaultManager] fileExistsAtPath:scriptPath])
            return [scriptPath stringByStandardizingPath];

        return nil;
    }
}

// ===== Model card I/O =====

// Load a model card from a JSON file. Returns nil if file doesn't exist.
static NSDictionary *nf_model_card_load(const char *path) {
    @autoreleasepool {
        NSData *data = [NSData dataWithContentsOfFile:
                        [NSString stringWithUTF8String:path]];
        if (!data) return nil;

        NSError *err = nil;
        NSDictionary *dict = [NSJSONSerialization JSONObjectWithData:data
                                                            options:0
                                                              error:&err];
        if (err || ![dict isKindOfClass:[NSDictionary class]]) return nil;
        return dict;
    }
}

// Emit a model card JSON file's contents as NDJSON for the app.
static void nf_emit_model_card(const char *card_path) {
    NSDictionary *card = nf_model_card_load(card_path);
    if (!card) return;

    @autoreleasepool {
        NSError *err = nil;
        NSData *data = [NSJSONSerialization dataWithJSONObject:card
                                                      options:0
                                                        error:&err];
        if (err || !data) return;

        NSString *json = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
        fprintf(stdout, "{\"type\":\"model_card\",\"card\":%s}\n", [json UTF8String]);
        fflush(stdout);
    }
}
