// config.h — NeuralForge CLI configuration
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>
#include <errno.h>
#include <math.h>
#import <Foundation/Foundation.h>

// Safe numeric parsers with range validation
static int nf_safe_atoi(const char *str, int default_val, int min_val, int max_val) {
    if (!str || *str == '\0') return default_val;
    char *endptr;
    errno = 0;
    long val = strtol(str, &endptr, 10);
    if (*endptr != '\0' || errno == ERANGE || val < min_val || val > max_val) {
        fprintf(stderr, "Warning: invalid integer '%s', using default %d\n", str, default_val);
        return default_val;
    }
    return (int)val;
}

static float nf_safe_atof(const char *str, float default_val, float min_val, float max_val) {
    if (!str || *str == '\0') return default_val;
    char *endptr;
    errno = 0;
    double val = strtod(str, &endptr);
    if (*endptr != '\0' || errno == ERANGE || isnan(val) || isinf(val) ||
        (float)val < min_val || (float)val > max_val) {
        fprintf(stderr, "Warning: invalid float '%s', using default %g\n", str, default_val);
        return default_val;
    }
    return (float)val;
}

typedef struct {
    char model_path[PATH_MAX];
    char data_path[PATH_MAX];
    char ckpt_path[PATH_MAX];

    int   total_steps;
    float learning_rate;
    int   accum_steps;
    int   checkpoint_every;
    float grad_clip_norm;
    float beta1, beta2, eps;

    bool  use_ane_extras;
    int   max_compiles;
    bool  resume;
    int   seed;
    bool  json_output;       // true = JSON to stdout (default for app)
} NFConfig;

static NFConfig nf_config_defaults(void) {
    NFConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    strlcpy(cfg.model_path, "stories110M.bin", PATH_MAX);
    strlcpy(cfg.data_path, "tinystories_data00.bin", PATH_MAX);
    strlcpy(cfg.ckpt_path, "neuralforge_ckpt.bin", PATH_MAX);
    cfg.total_steps = 10000;
    cfg.learning_rate = 3e-4f;
    cfg.accum_steps = 10;
    cfg.checkpoint_every = 100;
    cfg.grad_clip_norm = 1.0f;
    cfg.beta1 = 0.9f;
    cfg.beta2 = 0.999f;
    cfg.eps = 1e-8f;
    cfg.use_ane_extras = true;
    cfg.max_compiles = 100;
    cfg.resume = false;
    cfg.seed = 42;
    cfg.json_output = true;
    return cfg;
}

// Load config from a JSON file, overlay onto existing config
static void nf_config_from_json(NFConfig *cfg, const char *path) {
    @autoreleasepool {
        NSData *data = [NSData dataWithContentsOfFile:
                        [NSString stringWithUTF8String:path]];
        if (!data) {
            fprintf(stderr, "Warning: cannot read config file: %s\n", path);
            return;
        }
        NSError *err = nil;
        NSDictionary *json = [NSJSONSerialization JSONObjectWithData:data
                              options:0 error:&err];
        if (!json || err) {
            fprintf(stderr, "Warning: invalid JSON in config file: %s\n", path);
            return;
        }

        if (json[@"model"] && [json[@"model"] isKindOfClass:[NSString class]])
            strlcpy(cfg->model_path, [json[@"model"] UTF8String], PATH_MAX);
        if (json[@"data"] && [json[@"data"] isKindOfClass:[NSString class]])
            strlcpy(cfg->data_path, [json[@"data"] UTF8String], PATH_MAX);
        if (json[@"checkpoint"] && [json[@"checkpoint"] isKindOfClass:[NSString class]])
            strlcpy(cfg->ckpt_path, [json[@"checkpoint"] UTF8String], PATH_MAX);
        // Validated numeric ranges (same bounds as CLI args)
        if (json[@"steps"]) {
            int v = [json[@"steps"] intValue];
            if (v >= 1 && v <= 10000000) cfg->total_steps = v;
        }
        if (json[@"learning_rate"]) {
            float v = [json[@"learning_rate"] floatValue];
            if (v >= 1e-8f && v <= 10.0f && !isnan(v)) cfg->learning_rate = v;
        }
        if (json[@"accum_steps"]) {
            int v = [json[@"accum_steps"] intValue];
            if (v >= 1 && v <= 10000) cfg->accum_steps = v;
        }
        if (json[@"checkpoint_every"]) {
            int v = [json[@"checkpoint_every"] intValue];
            if (v >= 1 && v <= 1000000) cfg->checkpoint_every = v;
        }
        if (json[@"grad_clip_norm"]) {
            float v = [json[@"grad_clip_norm"] floatValue];
            if (v >= 0.0f && v <= 1000.0f && !isnan(v)) cfg->grad_clip_norm = v;
        }
        if (json[@"beta1"]) {
            float v = [json[@"beta1"] floatValue];
            if (v >= 0.0f && v <= 1.0f && !isnan(v)) cfg->beta1 = v;
        }
        if (json[@"beta2"]) {
            float v = [json[@"beta2"] floatValue];
            if (v >= 0.0f && v <= 1.0f && !isnan(v)) cfg->beta2 = v;
        }
        if (json[@"eps"]) {
            float v = [json[@"eps"] floatValue];
            if (v >= 1e-30f && v <= 1.0f && !isnan(v)) cfg->eps = v;
        }
        if (json[@"seed"]) {
            int v = [json[@"seed"] intValue];
            if (v >= 0) cfg->seed = v;
        }
        if (json[@"max_compiles"]) {
            int v = [json[@"max_compiles"] intValue];
            if (v >= 1 && v <= 10000) cfg->max_compiles = v;
        }
        if (json[@"use_ane_extras"])  cfg->use_ane_extras  = [json[@"use_ane_extras"] boolValue];
        if (json[@"resume"])          cfg->resume          = [json[@"resume"] boolValue];
    }
}

static NFConfig nf_config_from_args(int argc, char **argv) {
    NFConfig cfg = nf_config_defaults();

    // First pass: find --config and load JSON base
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--config") == 0 && i+1 < argc) {
            nf_config_from_json(&cfg, argv[++i]);
            break;
        }
    }

    // Second pass: CLI args override JSON values
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i+1 < argc)
            strlcpy(cfg.model_path, argv[++i], PATH_MAX);
        else if (strcmp(argv[i], "--data") == 0 && i+1 < argc)
            strlcpy(cfg.data_path, argv[++i], PATH_MAX);
        else if (strcmp(argv[i], "--ckpt") == 0 && i+1 < argc)
            strlcpy(cfg.ckpt_path, argv[++i], PATH_MAX);
        else if (strcmp(argv[i], "--steps") == 0 && i+1 < argc)
            cfg.total_steps = nf_safe_atoi(argv[++i], 10000, 1, 10000000);
        else if (strcmp(argv[i], "--lr") == 0 && i+1 < argc)
            cfg.learning_rate = nf_safe_atof(argv[++i], 3e-4f, 1e-8f, 10.0f);
        else if (strcmp(argv[i], "--accum") == 0 && i+1 < argc)
            cfg.accum_steps = nf_safe_atoi(argv[++i], 10, 1, 10000);
        else if (strcmp(argv[i], "--ckpt-every") == 0 && i+1 < argc)
            cfg.checkpoint_every = nf_safe_atoi(argv[++i], 100, 1, 1000000);
        else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc)
            cfg.seed = nf_safe_atoi(argv[++i], 42, 0, INT_MAX - 1);
        else if (strcmp(argv[i], "--beta1") == 0 && i+1 < argc)
            cfg.beta1 = nf_safe_atof(argv[++i], 0.9f, 0.0f, 1.0f);
        else if (strcmp(argv[i], "--beta2") == 0 && i+1 < argc)
            cfg.beta2 = nf_safe_atof(argv[++i], 0.999f, 0.0f, 1.0f);
        else if (strcmp(argv[i], "--eps") == 0 && i+1 < argc)
            cfg.eps = nf_safe_atof(argv[++i], 1e-8f, 1e-30f, 1.0f);
        else if (strcmp(argv[i], "--grad-clip") == 0 && i+1 < argc)
            cfg.grad_clip_norm = nf_safe_atof(argv[++i], 1.0f, 0.0f, 1000.0f);
        else if (strcmp(argv[i], "--max-compiles") == 0 && i+1 < argc)
            cfg.max_compiles = nf_safe_atoi(argv[++i], 100, 1, 10000);
        else if (strcmp(argv[i], "--resume") == 0)
            cfg.resume = true;
        else if (strcmp(argv[i], "--no-ane-extras") == 0)
            cfg.use_ane_extras = false;
        else if (strcmp(argv[i], "--no-json") == 0)
            cfg.json_output = false;
        else if (strcmp(argv[i], "--config") == 0 && i+1 < argc)
            i++;  // already handled in first pass
    }
    return cfg;
}
