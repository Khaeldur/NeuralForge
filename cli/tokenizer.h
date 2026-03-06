// tokenizer.h — BPE tokenizer for llama2.c tokenizer.bin format
// Format: max_token_length(i32) then for each vocab token:
//   score(f32), len(i32), bytes(len bytes)
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

typedef struct {
    char **vocab;
    float *scores;
    int vocab_size;
    int max_token_length;
    // Sorted index for merge lookups
    int *sorted_indices;
} NFTokenizer;

static int tok_compare(void *ctx, const void *a, const void *b) {
    NFTokenizer *t = (NFTokenizer *)ctx;
    return strcmp(t->vocab[*(const int *)a], t->vocab[*(const int *)b]);
}

static bool nf_tokenizer_load(NFTokenizer *t, const char *path, int vocab_size) {
    if (!t || !path || vocab_size <= 0 || vocab_size > 1000000) return false;
    FILE *f = fopen(path, "rb");
    if (!f) return false;

    memset(t, 0, sizeof(*t));
    t->vocab_size = vocab_size;
    t->vocab = (char **)calloc(vocab_size, sizeof(char *));
    t->scores = (float *)malloc(vocab_size * sizeof(float));
    if (!t->vocab || !t->scores) { fclose(f); free(t->vocab); free(t->scores); return false; }

    if (fread(&t->max_token_length, sizeof(int), 1, f) != 1) { fclose(f); return false; }
    // Validate max_token_length to prevent absurd allocations
    if (t->max_token_length <= 0 || t->max_token_length > 65536) { fclose(f); return false; }

    for (int i = 0; i < vocab_size; i++) {
        float score;
        int len;
        if (fread(&score, sizeof(float), 1, f) != 1) { fclose(f); return false; }
        if (fread(&len, sizeof(int), 1, f) != 1) { fclose(f); return false; }
        // Validate token length against max and sanity check
        if (len < 0 || len > t->max_token_length) { fclose(f); return false; }
        t->scores[i] = score;
        t->vocab[i] = (char *)malloc(len + 1);
        if (!t->vocab[i]) { fclose(f); return false; }
        if (fread(t->vocab[i], 1, len, f) != (size_t)len) { fclose(f); return false; }
        t->vocab[i][len] = '\0';
    }
    fclose(f);

    // Build sorted index for binary search
    t->sorted_indices = (int *)malloc(vocab_size * sizeof(int));
    if (!t->sorted_indices) return false;
    for (int i = 0; i < vocab_size; i++) t->sorted_indices[i] = i;
    qsort_r(t->sorted_indices, vocab_size, sizeof(int), t, tok_compare);

    return true;
}

static int nf_tokenizer_lookup(NFTokenizer *t, const char *str) {
    // Binary search on sorted vocabulary
    int lo = 0, hi = t->vocab_size - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int cmp = strcmp(str, t->vocab[t->sorted_indices[mid]]);
        if (cmp == 0) return t->sorted_indices[mid];
        if (cmp < 0) hi = mid - 1; else lo = mid + 1;
    }
    return -1;
}

// Encode a string into token IDs using BPE
// Returns number of tokens written to output
static int nf_tokenizer_encode(NFTokenizer *t, const char *text, uint16_t *output, int max_tokens) {
    if (!text || !*text) return 0;

    // Start with character-level tokens
    int n_tokens = 0;
    const char *p = text;
    while (*p && n_tokens < max_tokens) {
        // Try to find this single character as a token
        char single[2] = { *p, '\0' };
        int id = nf_tokenizer_lookup(t, single);
        if (id >= 0) {
            output[n_tokens++] = (uint16_t)id;
        }
        // Skip unknown bytes (shouldn't happen with llama2 vocab but be safe)
        p++;
    }

    // BPE merge loop — iteratively merge the highest-scoring pair
    char *merge_buf = (char *)malloc(t->max_token_length * 2 + 1);
    if (!merge_buf) return n_tokens;
    while (n_tokens > 1) {
        float best_score = -1e30f;
        int best_idx = -1, best_id = -1;

        for (int i = 0; i < n_tokens - 1; i++) {
            snprintf(merge_buf, t->max_token_length * 2 + 1, "%s%s",
                     t->vocab[output[i]], t->vocab[output[i + 1]]);
            int id = nf_tokenizer_lookup(t, merge_buf);
            if (id >= 0 && t->scores[id] > best_score) {
                best_score = t->scores[id];
                best_idx = i;
                best_id = id;
            }
        }

        if (best_idx < 0) break;  // No more merges possible

        // Merge: replace pair at best_idx with merged token
        output[best_idx] = (uint16_t)best_id;
        // Shift remaining tokens left
        for (int i = best_idx + 1; i < n_tokens - 1; i++) {
            output[i] = output[i + 1];
        }
        n_tokens--;
    }
    free(merge_buf);

    return n_tokens;
}

static void nf_tokenizer_free(NFTokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab);
    free(t->scores);
    free(t->sorted_indices);
    memset(t, 0, sizeof(*t));
}
