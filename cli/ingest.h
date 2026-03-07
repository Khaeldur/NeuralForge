// ingest.h — Document ingestion pipeline for NeuralForge
// Scans a source directory for documents (TXT, PDF, DOCX, MD, CSV, code files),
// extracts text, tokenizes, and writes output as token shards with manifest tracking.
//
// Supported input formats:
//   Plain text: .txt, .md, .csv, .json, .py, .js, .c, .m, .h, .swift, .rs, .go, .java, .ts, .tsx, .jsx
//   Rich text:  .pdf (via PDFKit), .docx (via macOS textutil)
//
// Output:
//   Shard files: shard_000.bin, shard_001.bin, ... (raw uint16_t token arrays)
//   Manifest:    manifest.json (tracks shards, processed files, timestamps)
//
// Usage:
//   neuralforge ingest --source <dir> --output <dir> --tokenizer tokenizer.bin
//     [--max-shard-mb 50] [--manifest manifest.json] [--incremental]
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <limits.h>
#import <Foundation/Foundation.h>
#import <Quartz/Quartz.h>

// ===== Supported extensions =====

static bool nf_ingest_is_supported(const char *ext) {
    if (!ext) return false;
    static const char *supported[] = {
        "txt", "md", "csv", "json",
        "py", "js", "c", "m", "h", "swift", "rs", "go", "java",
        "ts", "tsx", "jsx",
        "pdf", "docx", NULL
    };
    for (int i = 0; supported[i]; i++) {
        if (strcasecmp(ext, supported[i]) == 0) return true;
    }
    return false;
}

// ===== Text extraction =====

// Extract text from a PDF using PDFKit.
// Returns a malloc'd C string, or NULL on failure.
static char *nf_extract_pdf(const char *path) {
    @autoreleasepool {
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
        PDFDocument *doc = [[PDFDocument alloc] initWithURL:url];
        if (!doc || doc.pageCount == 0) return NULL;

        NSMutableString *text = [NSMutableString stringWithCapacity:4096];
        for (NSUInteger i = 0; i < doc.pageCount; i++) {
            PDFPage *page = [doc pageAtIndex:i];
            NSString *pageText = [page string];
            if (pageText.length > 0) {
                [text appendString:pageText];
                [text appendString:@"\n\n"];
            }
        }
        if (text.length == 0) return NULL;
        return strdup([text UTF8String]);
    }
}

// Extract text from a DOCX using macOS built-in textutil command.
// Returns a malloc'd C string, or NULL on failure.
static char *nf_extract_docx(const char *path) {
    @autoreleasepool {
        NSTask *task = [[NSTask alloc] init];
        task.executableURL = [NSURL fileURLWithPath:@"/usr/bin/textutil"];
        task.arguments = @[@"-convert", @"txt", @"-stdout",
                           [NSString stringWithUTF8String:path]];
        NSPipe *pipe = [NSPipe pipe];
        task.standardOutput = pipe;
        task.standardError = [NSFileHandle fileHandleWithNullDevice];

        NSError *err = nil;
        [task launchAndReturnError:&err];
        if (err) return NULL;
        [task waitUntilExit];
        if (task.terminationStatus != 0) return NULL;

        NSData *data = [pipe.fileHandleForReading readDataToEndOfFile];
        if (data.length == 0) return NULL;

        NSString *text = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
        if (!text || text.length == 0) return NULL;
        return strdup([text UTF8String]);
    }
}

// Extract text from a plain text file.
// Returns a malloc'd C string, or NULL on failure.
static char *nf_extract_plain(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    if (size <= 0 || size > (long)2e9) { fclose(f); return NULL; }
    fseek(f, 0, SEEK_SET);

    char *text = (char *)malloc(size + 1);
    if (!text) { fclose(f); return NULL; }
    size_t nread = fread(text, 1, size, f);
    text[nread] = '\0';
    fclose(f);

    if (nread == 0) { free(text); return NULL; }
    return text;
}

// Extract text from any supported file format.
// Returns a malloc'd C string (caller must free), or NULL on error.
static char *nf_extract_text(const char *path) {
    @autoreleasepool {
        NSString *ext = [[NSString stringWithUTF8String:path] pathExtension];
        ext = [ext lowercaseString];

        if ([ext isEqualToString:@"pdf"]) return nf_extract_pdf(path);
        if ([ext isEqualToString:@"docx"]) return nf_extract_docx(path);
        return nf_extract_plain(path);
    }
}

// ===== Directory scanning =====

// Get file modification time as ISO 8601 string.
// Returns a static-duration string valid until next call, or NULL on error.
static const char *nf_file_mtime_iso(const char *path) {
    static char buf[32];
    struct stat st;
    if (stat(path, &st) != 0) return NULL;
    struct tm *tm = gmtime(&st.st_mtime);
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", tm);
    return buf;
}

// Scan a directory recursively for supported document files.
// Returns an NSArray<NSString *> of absolute paths, sorted alphabetically.
static NSArray<NSString *> *nf_scan_source_dir(const char *dir_path) {
    @autoreleasepool {
        NSFileManager *fm = [NSFileManager defaultManager];
        NSString *dir = [NSString stringWithUTF8String:dir_path];

        if (![fm fileExistsAtPath:dir]) return @[];

        NSMutableArray<NSString *> *results = [NSMutableArray array];

        NSDirectoryEnumerator *enumerator =
            [fm enumeratorAtURL:[NSURL fileURLWithPath:dir]
     includingPropertiesForKeys:@[NSURLIsRegularFileKey]
                        options:NSDirectoryEnumerationSkipsHiddenFiles
                   errorHandler:nil];

        for (NSURL *url in enumerator) {
            NSNumber *isFile = nil;
            [url getResourceValue:&isFile forKey:NSURLIsRegularFileKey error:nil];
            if (![isFile boolValue]) continue;

            const char *ext = [url.pathExtension UTF8String];
            if (nf_ingest_is_supported(ext)) {
                [results addObject:url.path];
            }
        }

        [results sortUsingSelector:@selector(compare:)];
        return [results copy];
    }
}

// ===== Manifest I/O =====

// Load a manifest from a JSON file. Returns nil if file doesn't exist or is invalid.
static NSDictionary *nf_manifest_load(const char *path) {
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

// Save a manifest dictionary to a JSON file. Returns true on success.
static bool nf_manifest_save(NSDictionary *manifest, const char *path) {
    @autoreleasepool {
        NSError *err = nil;
        NSData *data = [NSJSONSerialization dataWithJSONObject:manifest
                                                      options:NSJSONWritingPrettyPrinted
                                                        error:&err];
        if (err || !data) return false;
        return [data writeToFile:[NSString stringWithUTF8String:path]
                      atomically:YES];
    }
}

// Build a new manifest dictionary from ingestion results.
// shards: array of {path, tokens, bytes, source_files} dictionaries
// processed: dictionary of {filename: {mtime, tokens}} entries
static NSDictionary *nf_manifest_build(NSArray *shards,
                                        NSDictionary *processed,
                                        const char *tokenizer_path,
                                        int vocab_size, int total_tokens) {
    @autoreleasepool {
        char timestamp[32];
        time_t now = time(NULL);
        struct tm *tm = gmtime(&now);
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", tm);

        return @{
            @"version": @1,
            @"last_run": [NSString stringWithUTF8String:timestamp],
            @"tokenizer": [NSString stringWithUTF8String:tokenizer_path],
            @"vocab_size": @(vocab_size),
            @"shards": shards ?: @[],
            @"total_tokens": @(total_tokens),
            @"processed_files": processed ?: @{}
        };
    }
}

// ===== Shard writing =====

// Write a shard file containing raw uint16_t tokens.
// Returns bytes written (count * 2), or 0 on error.
static size_t nf_write_shard(const uint16_t *tokens, int count, const char *path) {
    if (!tokens || count <= 0 || !path) return 0;
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    size_t written = fwrite(tokens, sizeof(uint16_t), count, f);
    fclose(f);
    return written * sizeof(uint16_t);
}

// Generate a shard filename: shard_000.bin, shard_001.bin, ...
static void nf_shard_name(int index, char *buf, size_t buflen) {
    snprintf(buf, buflen, "shard_%03d.bin", index);
}

// ===== Incremental check =====

// Check if a file should be processed in incremental mode.
// Returns true if the file is new or modified since the last manifest entry.
static bool nf_should_process(const char *file_path, NSDictionary *manifest) {
    if (!manifest) return true;  // No manifest = process everything

    @autoreleasepool {
        NSDictionary *processed = manifest[@"processed_files"];
        if (!processed) return true;

        NSString *filename = [[NSString stringWithUTF8String:file_path] lastPathComponent];
        NSDictionary *entry = processed[filename];
        if (!entry) return true;  // File not in manifest

        // Compare modification times
        NSString *recorded_mtime = entry[@"mtime"];
        if (!recorded_mtime) return true;

        const char *current_mtime = nf_file_mtime_iso(file_path);
        if (!current_mtime) return true;

        return ![recorded_mtime isEqualToString:
                 [NSString stringWithUTF8String:current_mtime]];
    }
}
