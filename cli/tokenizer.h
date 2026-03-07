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

// --- Priority queue (max-heap) for fast BPE merging ---

typedef struct {
    float score;
    int pos;         // index of left token in node array
    uint16_t left_id;  // expected left token (staleness check)
    uint16_t right_id; // expected right token (staleness check)
} NFMerge;

typedef struct {
    uint16_t token_id;
    int prev;  // -1 = head
    int next;  // -1 = tail, -2 = deleted
} NFTokNode;

static void nf_heap_push(NFMerge *heap, int *size, NFMerge m) {
    int i = (*size)++;
    heap[i] = m;
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (heap[parent].score >= heap[i].score) break;
        NFMerge tmp = heap[parent];
        heap[parent] = heap[i];
        heap[i] = tmp;
        i = parent;
    }
}

static NFMerge nf_heap_pop(NFMerge *heap, int *size) {
    NFMerge top = heap[0];
    heap[0] = heap[--(*size)];
    int i = 0;
    for (;;) {
        int best = i, left = 2*i+1, right = 2*i+2;
        if (left < *size && heap[left].score > heap[best].score) best = left;
        if (right < *size && heap[right].score > heap[best].score) best = right;
        if (best == i) break;
        NFMerge tmp = heap[i];
        heap[i] = heap[best];
        heap[best] = tmp;
        i = best;
    }
    return top;
}

// Try to add a merge candidate for adjacent nodes at positions left, right
static void nf_try_add_merge(NFTokenizer *t, NFTokNode *nodes, NFMerge *heap,
                             int *heap_size, int heap_cap, char *buf, int left, int right) {
    if (left < 0 || right < 0) return;
    snprintf(buf, t->max_token_length * 2 + 1, "%s%s",
             t->vocab[nodes[left].token_id], t->vocab[nodes[right].token_id]);
    int merge_id = nf_tokenizer_lookup(t, buf);
    if (merge_id >= 0 && *heap_size < heap_cap) {
        NFMerge m = { t->scores[merge_id], left, nodes[left].token_id, nodes[right].token_id };
        nf_heap_push(heap, heap_size, m);
    }
}

// Encode a string into token IDs using BPE with priority queue
// O(n log n) instead of O(n^2) — handles megabyte-scale inputs
static int nf_tokenizer_encode(NFTokenizer *t, const char *text, uint16_t *output, int max_tokens) {
    if (!text || !*text) return 0;

    int text_len = (int)strlen(text);

    // Phase 1: Character-level tokenization into doubly-linked list
    NFTokNode *nodes = (NFTokNode *)malloc(text_len * sizeof(NFTokNode));
    if (!nodes) return 0;

    int n_nodes = 0;
    const char *p = text;
    while (*p && n_nodes < text_len) {
        char single[2] = { *p, '\0' };
        int id = nf_tokenizer_lookup(t, single);
        if (id >= 0) {
            nodes[n_nodes].token_id = (uint16_t)id;
            nodes[n_nodes].prev = n_nodes - 1;
            nodes[n_nodes].next = n_nodes + 1;
            n_nodes++;
        }
        p++;
    }
    if (n_nodes == 0) { free(nodes); return 0; }
    nodes[0].prev = -1;
    nodes[n_nodes - 1].next = -1;

    // Phase 2: Build initial merge heap from all adjacent pairs
    int heap_cap = (n_nodes < 4) ? 16 : n_nodes * 3;
    NFMerge *heap = (NFMerge *)malloc(heap_cap * sizeof(NFMerge));
    if (!heap) {
        // Fallback: just return character tokens
        int n = (n_nodes < max_tokens) ? n_nodes : max_tokens;
        for (int i = 0; i < n; i++) output[i] = nodes[i].token_id;
        free(nodes);
        return n;
    }
    int heap_size = 0;

    char *merge_buf = (char *)malloc(t->max_token_length * 2 + 1);
    if (!merge_buf) { free(nodes); free(heap); return 0; }

    for (int i = 0; i < n_nodes - 1; i++) {
        nf_try_add_merge(t, nodes, heap, &heap_size, heap_cap, merge_buf, i, i + 1);
    }

    // Phase 3: Greedily apply best merges via heap
    int active = n_nodes;
    while (heap_size > 0 && active > 1) {
        NFMerge best = nf_heap_pop(heap, &heap_size);
        int pos = best.pos;

        // Staleness check: skip if tokens changed since this merge was enqueued
        if (nodes[pos].next == -2) continue;        // left node was deleted
        if (nodes[pos].token_id != best.left_id) continue;  // left token changed
        if (nodes[pos].next < 0) continue;           // no right neighbor
        int rpos = nodes[pos].next;
        if (nodes[rpos].token_id != best.right_id) continue; // right token changed

        // Verify merge is still valid (redundant but safe)
        snprintf(merge_buf, t->max_token_length * 2 + 1, "%s%s",
                 t->vocab[nodes[pos].token_id], t->vocab[nodes[rpos].token_id]);
        int merge_id = nf_tokenizer_lookup(t, merge_buf);
        if (merge_id < 0) continue;

        // Apply merge: update left node, unlink right node
        nodes[pos].token_id = (uint16_t)merge_id;
        nodes[pos].next = nodes[rpos].next;
        if (nodes[rpos].next >= 0) {
            nodes[nodes[rpos].next].prev = pos;
        }
        nodes[rpos].next = -2;  // mark deleted
        nodes[rpos].prev = -2;
        active--;

        // Grow heap if needed before adding new candidates
        if (heap_size + 2 >= heap_cap) {
            heap_cap *= 2;
            NFMerge *new_heap = (NFMerge *)realloc(heap, heap_cap * sizeof(NFMerge));
            if (!new_heap) break;  // out of memory, stop merging
            heap = new_heap;
        }

        // Add new merge candidates for the updated neighbors
        if (nodes[pos].prev >= 0) {
            nf_try_add_merge(t, nodes, heap, &heap_size, heap_cap, merge_buf,
                             nodes[pos].prev, pos);
        }
        if (nodes[pos].next >= 0) {
            nf_try_add_merge(t, nodes, heap, &heap_size, heap_cap, merge_buf,
                             pos, nodes[pos].next);
        }
    }

    // Phase 4: Collect results by walking the linked list from head
    int n_output = 0;
    int cur = 0;
    // Find head (first node with prev == -1)
    for (int i = 0; i < n_nodes; i++) {
        if (nodes[i].prev == -1 && nodes[i].next != -2) { cur = i; break; }
    }
    while (cur >= 0 && n_output < max_tokens) {
        output[n_output++] = nodes[cur].token_id;
        cur = nodes[cur].next;
    }

    free(merge_buf);
    free(heap);
    free(nodes);
    return n_output;
}

// Decode a single token ID back to its string
// Returns NULL for out-of-range IDs
static const char *nf_tokenizer_decode(NFTokenizer *t, int token_id) {
    if (!t || token_id < 0 || token_id >= t->vocab_size) return NULL;
    return t->vocab[token_id];
}

static void nf_tokenizer_free(NFTokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab);
    free(t->scores);
    free(t->sorted_indices);
    memset(t, 0, sizeof(*t));
}
