// audit.h — Append-only audit log for NeuralForge
// Writes JSONL entries to ~/Library/Logs/NeuralForge/audit.jsonl
// Each entry includes a SHA-256 hash chain for tamper detection.
//
// Events logged:
//   training_start, training_stop, checkpoint_save, export,
//   generate_start, generate_done, config_used
//
// Usage:
//   nf_audit_init();                          // call once at startup
//   nf_audit_log("training_start", details);  // log an event
//   nf_audit_close();                         // flush and close
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <CommonCrypto/CommonDigest.h>

// Maximum size for a single audit detail string
#define NF_AUDIT_DETAIL_MAX 4096
// Maximum size for the complete JSON line
#define NF_AUDIT_LINE_MAX   8192

typedef struct {
    FILE *fp;
    char path[1024];
    char prev_hash[65];  // hex-encoded SHA-256 of previous entry (64 + NUL)
    int  seq;            // sequence number
    bool initialized;
} NFAuditLog;

// Global audit log instance
static NFAuditLog g_audit = {0};

// Compute SHA-256 hex digest of a string
static void nf_sha256_hex(const char *input, size_t len, char *output) {
    unsigned char hash[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(input, (CC_LONG)len, hash);
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) {
        snprintf(output + i * 2, 3, "%02x", hash[i]);
    }
    output[CC_SHA256_DIGEST_LENGTH * 2] = '\0';
}

// Get ISO 8601 timestamp
static void nf_audit_timestamp(char *buf, size_t buflen) {
    time_t now = time(NULL);
    struct tm *tm = gmtime(&now);
    strftime(buf, buflen, "%Y-%m-%dT%H:%M:%SZ", tm);
}

// Get current user name (for "who" field)
static const char *nf_audit_user(void) {
    const char *user = getenv("USER");
    if (!user) user = getenv("LOGNAME");
    if (!user) user = "unknown";
    return user;
}

// Initialize the audit log — creates directory and opens file
static bool nf_audit_init(void) {
    if (g_audit.initialized) return true;

    // Build path: ~/Library/Logs/NeuralForge/audit.jsonl
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    snprintf(g_audit.path, sizeof(g_audit.path),
             "%s/Library/Logs/NeuralForge/audit.jsonl", home);

    // Create directory if needed
    char dir[1024];
    snprintf(dir, sizeof(dir), "%s/Library/Logs/NeuralForge", home);
    mkdir(dir, 0755);  // Ignore error if exists

    // Open for append
    g_audit.fp = fopen(g_audit.path, "a");
    if (!g_audit.fp) {
        fprintf(stderr, "Warning: cannot open audit log %s\n", g_audit.path);
        return false;
    }

    // Read last hash from existing file for chain continuity
    strlcpy(g_audit.prev_hash,
            "0000000000000000000000000000000000000000000000000000000000000000",
            sizeof(g_audit.prev_hash));
    g_audit.seq = 0;

    // Scan existing file to find last entry's hash and seq
    FILE *rf = fopen(g_audit.path, "r");
    if (rf) {
        char line[NF_AUDIT_LINE_MAX];
        while (fgets(line, sizeof(line), rf)) {
            // Extract "hash":"..." field
            const char *hp = strstr(line, "\"hash\":\"");
            if (hp) {
                hp += 8;  // skip "hash":"
                if (strlen(hp) >= 64) {
                    memcpy(g_audit.prev_hash, hp, 64);
                    g_audit.prev_hash[64] = '\0';
                }
            }
            // Extract "seq": field
            const char *sp = strstr(line, "\"seq\":");
            if (sp) {
                sp += 6;
                g_audit.seq = atoi(sp);
            }
        }
        fclose(rf);
    }

    g_audit.initialized = true;
    return true;
}

// Log an audit event with JSON details
// event: event type string (e.g., "training_start")
// details: JSON object string (without outer braces), e.g.:
//   "\"model\":\"stories110M.bin\",\"steps\":1000"
//   Pass NULL or "" for no extra details.
static void nf_audit_log(const char *event, const char *details) {
    if (!g_audit.initialized) {
        if (!nf_audit_init()) return;
    }
    if (!g_audit.fp || !event) return;

    char timestamp[32];
    nf_audit_timestamp(timestamp, sizeof(timestamp));

    const char *user = nf_audit_user();
    int seq = ++g_audit.seq;

    // Build the entry content (everything except the hash field)
    // We compute hash over: prev_hash + seq + timestamp + event + user + details
    char content[NF_AUDIT_LINE_MAX];
    int clen;
    if (details && details[0]) {
        clen = snprintf(content, sizeof(content),
                        "%s|%d|%s|%s|%s|%s",
                        g_audit.prev_hash, seq, timestamp, event, user, details);
    } else {
        clen = snprintf(content, sizeof(content),
                        "%s|%d|%s|%s|%s",
                        g_audit.prev_hash, seq, timestamp, event, user);
    }
    if (clen < 0 || clen >= (int)sizeof(content)) return;

    // Compute SHA-256 hash of content
    char hash[65];
    nf_sha256_hex(content, (size_t)clen, hash);

    // Write JSON line
    if (details && details[0]) {
        fprintf(g_audit.fp,
                "{\"seq\":%d,\"time\":\"%s\",\"event\":\"%s\","
                "\"user\":\"%s\",%s,"
                "\"prev_hash\":\"%s\",\"hash\":\"%s\"}\n",
                seq, timestamp, event, user, details,
                g_audit.prev_hash, hash);
    } else {
        fprintf(g_audit.fp,
                "{\"seq\":%d,\"time\":\"%s\",\"event\":\"%s\","
                "\"user\":\"%s\","
                "\"prev_hash\":\"%s\",\"hash\":\"%s\"}\n",
                seq, timestamp, event, user,
                g_audit.prev_hash, hash);
    }
    fflush(g_audit.fp);

    // Update chain
    strlcpy(g_audit.prev_hash, hash, sizeof(g_audit.prev_hash));
}

// Close the audit log
static void nf_audit_close(void) {
    if (g_audit.fp) {
        fflush(g_audit.fp);
        fclose(g_audit.fp);
        g_audit.fp = NULL;
    }
    g_audit.initialized = false;
}

// ===== Convenience logging functions =====

// Log training start with full config
static void nf_audit_training_start(const char *model, const char *data,
                                     int steps, float lr, int accum,
                                     int lora_rank, float lora_alpha) {
    char details[NF_AUDIT_DETAIL_MAX];
    if (lora_rank > 0) {
        snprintf(details, sizeof(details),
                 "\"model\":\"%s\",\"data\":\"%s\","
                 "\"steps\":%d,\"lr\":%.6e,\"accum\":%d,"
                 "\"lora_rank\":%d,\"lora_alpha\":%.1f",
                 model, data, steps, lr, accum, lora_rank, lora_alpha);
    } else {
        snprintf(details, sizeof(details),
                 "\"model\":\"%s\",\"data\":\"%s\","
                 "\"steps\":%d,\"lr\":%.6e,\"accum\":%d",
                 model, data, steps, lr, accum);
    }
    nf_audit_log("training_start", details);
}

// Log training stop with results
static void nf_audit_training_stop(int steps_done, float final_loss,
                                    double total_time_s, const char *reason) {
    char details[NF_AUDIT_DETAIL_MAX];
    snprintf(details, sizeof(details),
             "\"steps_done\":%d,\"final_loss\":%.6f,"
             "\"total_time_s\":%.2f,\"reason\":\"%s\"",
             steps_done, final_loss, total_time_s,
             reason ? reason : "completed");
    nf_audit_log("training_stop", details);
}

// Log checkpoint save
static void nf_audit_checkpoint(const char *path, int step, float loss) {
    char details[NF_AUDIT_DETAIL_MAX];
    snprintf(details, sizeof(details),
             "\"path\":\"%s\",\"step\":%d,\"loss\":%.6f",
             path, step, loss);
    nf_audit_log("checkpoint_save", details);
}

// Log model export
static void nf_audit_export(const char *input_path, const char *output_path,
                             const char *format) {
    char details[NF_AUDIT_DETAIL_MAX];
    snprintf(details, sizeof(details),
             "\"input\":\"%s\",\"output\":\"%s\",\"format\":\"%s\"",
             input_path, output_path, format);
    nf_audit_log("export", details);
}

// Log text generation start
static void nf_audit_generate_start(const char *model, int max_tokens,
                                     float temperature, float top_p) {
    char details[NF_AUDIT_DETAIL_MAX];
    snprintf(details, sizeof(details),
             "\"model\":\"%s\",\"max_tokens\":%d,"
             "\"temperature\":%.2f,\"top_p\":%.2f",
             model, max_tokens, temperature, top_p);
    nf_audit_log("generate_start", details);
}

// Log text generation done
static void nf_audit_generate_done(int tokens_generated, double total_ms) {
    char details[NF_AUDIT_DETAIL_MAX];
    snprintf(details, sizeof(details),
             "\"tokens\":%d,\"total_ms\":%.1f",
             tokens_generated, total_ms);
    nf_audit_log("generate_done", details);
}

// Verify the hash chain integrity of the audit log
// Returns: number of valid entries, or -1 on error
// Sets *first_bad to the seq number of first broken entry (0 if all valid)
static int nf_audit_verify(const char *path, int *first_bad) {
    if (first_bad) *first_bad = 0;

    FILE *f = fopen(path, "r");
    if (!f) return -1;

    char prev_hash[65];
    strlcpy(prev_hash,
            "0000000000000000000000000000000000000000000000000000000000000000",
            sizeof(prev_hash));

    int count = 0;
    char line[NF_AUDIT_LINE_MAX];

    while (fgets(line, sizeof(line), f)) {
        // Strip trailing newline
        size_t len = strlen(line);
        if (len > 0 && line[len - 1] == '\n') line[--len] = '\0';
        if (len == 0) continue;

        // Extract fields from JSON
        int seq = 0;
        char timestamp[32] = {0}, event[64] = {0}, user[64] = {0};
        char entry_prev_hash[65] = {0}, entry_hash[65] = {0};

        // Parse seq
        const char *sp = strstr(line, "\"seq\":");
        if (sp) seq = atoi(sp + 6);

        // Parse time
        const char *tp = strstr(line, "\"time\":\"");
        if (tp) {
            tp += 8;
            const char *te = strchr(tp, '"');
            if (te && te - tp < 32) { memcpy(timestamp, tp, te - tp); timestamp[te - tp] = '\0'; }
        }

        // Parse event
        const char *ep = strstr(line, "\"event\":\"");
        if (ep) {
            ep += 9;
            const char *ee = strchr(ep, '"');
            if (ee && ee - ep < 64) { memcpy(event, ep, ee - ep); event[ee - ep] = '\0'; }
        }

        // Parse user
        const char *up = strstr(line, "\"user\":\"");
        if (up) {
            up += 8;
            const char *ue = strchr(up, '"');
            if (ue && ue - up < 64) { memcpy(user, up, ue - up); user[ue - up] = '\0'; }
        }

        // Parse prev_hash
        const char *php = strstr(line, "\"prev_hash\":\"");
        if (php) {
            php += 13;
            if (strlen(php) >= 64) { memcpy(entry_prev_hash, php, 64); entry_prev_hash[64] = '\0'; }
        }

        // Parse hash
        const char *hp = strstr(line, "\"hash\":\"");
        if (hp) {
            hp += 8;
            if (strlen(hp) >= 64) { memcpy(entry_hash, hp, 64); entry_hash[64] = '\0'; }
        }

        // Verify prev_hash matches our tracked chain
        if (strcmp(entry_prev_hash, prev_hash) != 0) {
            if (first_bad) *first_bad = seq;
            fclose(f);
            return count;
        }

        // Recompute hash and verify
        // We need to find the details portion — everything between user and prev_hash
        // For simplicity in verification, we extract fields and reconstruct content
        // Extract details: find content between "user":"..." and "prev_hash"
        char details[NF_AUDIT_DETAIL_MAX] = {0};
        // Find the comma after user field to get details
        const char *after_user = strstr(line, "\"user\":\"");
        if (after_user) {
            after_user += 8;
            const char *user_end = strchr(after_user, '"');
            if (user_end) {
                user_end++; // skip closing quote
                if (*user_end == ',') {
                    user_end++; // skip comma
                    const char *prev_hash_start = strstr(user_end, "\"prev_hash\":");
                    if (prev_hash_start && prev_hash_start > user_end) {
                        size_t dlen = (size_t)(prev_hash_start - user_end);
                        if (dlen > 0 && dlen < sizeof(details)) {
                            memcpy(details, user_end, dlen);
                            // Remove trailing comma
                            while (dlen > 0 && (details[dlen - 1] == ',' || details[dlen - 1] == ' '))
                                dlen--;
                            details[dlen] = '\0';
                        }
                    }
                }
            }
        }

        // Reconstruct content for hash verification
        char content[NF_AUDIT_LINE_MAX];
        int clen;
        if (details[0]) {
            clen = snprintf(content, sizeof(content),
                            "%s|%d|%s|%s|%s|%s",
                            entry_prev_hash, seq, timestamp, event, user, details);
        } else {
            clen = snprintf(content, sizeof(content),
                            "%s|%d|%s|%s|%s",
                            entry_prev_hash, seq, timestamp, event, user);
        }

        char computed_hash[65];
        nf_sha256_hex(content, (size_t)clen, computed_hash);

        if (strcmp(computed_hash, entry_hash) != 0) {
            if (first_bad) *first_bad = seq;
            fclose(f);
            return count;
        }

        strlcpy(prev_hash, entry_hash, sizeof(prev_hash));
        count++;
    }

    fclose(f);
    return count;
}
