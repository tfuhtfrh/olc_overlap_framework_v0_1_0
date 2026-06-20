#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BASE_COUNT 4
#define PI 3.141592653589793238462643383279502884

typedef struct {
    double re;
    double im;
} Complex;

typedef struct {
    int src;
    int dst;
    int overlap;
    int matches;
    double identity;
} Edge;

typedef struct {
    char **items;
    int count;
    int capacity;
} StringList;

typedef struct {
    Edge *items;
    int count;
    int capacity;
} EdgeList;

static Complex c_add(Complex a, Complex b) {
    Complex out = {a.re + b.re, a.im + b.im};
    return out;
}

static Complex c_sub(Complex a, Complex b) {
    Complex out = {a.re - b.re, a.im - b.im};
    return out;
}

static Complex c_mul(Complex a, Complex b) {
    Complex out = {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
    return out;
}

static int base_index(char ch) {
    switch (ch) {
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        case 'T': return 3;
        default: return -1;
    }
}

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int next_power_of_two(int value) {
    int out = 1;
    while (out < value) {
        out <<= 1;
    }
    return out;
}

static void fft(Complex *a, int n, int invert) {
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;

        if (i < j) {
            Complex tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }
    }

    for (int len = 2; len <= n; len <<= 1) {
        double angle = 2.0 * PI / (double)len * (invert ? -1.0 : 1.0);
        Complex wlen = {cos(angle), sin(angle)};

        for (int i = 0; i < n; i += len) {
            Complex w = {1.0, 0.0};
            for (int j = 0; j < len / 2; ++j) {
                Complex u = a[i + j];
                Complex v = c_mul(a[i + j + len / 2], w);
                a[i + j] = c_add(u, v);
                a[i + j + len / 2] = c_sub(u, v);
                w = c_mul(w, wlen);
            }
        }
    }

    if (invert) {
        for (int i = 0; i < n; ++i) {
            a[i].re /= (double)n;
            a[i].im /= (double)n;
        }
    }
}

static void die(const char *message) {
    fprintf(stderr, "Error: %s\n", message);
    exit(1);
}

static void *xmalloc(size_t size) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        die("out of memory");
    }
    return ptr;
}

static void *xcalloc(size_t count, size_t size) {
    void *ptr = calloc(count, size);
    if (ptr == NULL) {
        die("out of memory");
    }
    return ptr;
}

static void *xrealloc(void *ptr, size_t size) {
    void *out = realloc(ptr, size);
    if (out == NULL) {
        die("out of memory");
    }
    return out;
}

static char *xstrdup(const char *text) {
    size_t len = strlen(text);
    char *out = (char *)xmalloc(len + 1);
    memcpy(out, text, len + 1);
    return out;
}

static void string_list_init(StringList *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

static void string_list_push(StringList *list, char *item) {
    if (list->count == list->capacity) {
        int new_capacity = list->capacity == 0 ? 16 : list->capacity * 2;
        list->items = (char **)xrealloc(list->items, (size_t)new_capacity * sizeof(char *));
        list->capacity = new_capacity;
    }
    list->items[list->count++] = item;
}

static void string_list_free(StringList *list) {
    for (int i = 0; i < list->count; ++i) {
        free(list->items[i]);
    }
    free(list->items);
}

static void edge_list_init(EdgeList *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

static void edge_list_push(EdgeList *list, Edge edge) {
    if (list->count == list->capacity) {
        int new_capacity = list->capacity == 0 ? 64 : list->capacity * 2;
        list->items = (Edge *)xrealloc(list->items, (size_t)new_capacity * sizeof(Edge));
        list->capacity = new_capacity;
    }
    list->items[list->count++] = edge;
}

static void append_sequence_chunk(char **seq, int *len, int *cap, const char *chunk) {
    int chunk_len = (int)strlen(chunk);
    if (*len + chunk_len + 1 > *cap) {
        int new_cap = *cap == 0 ? 256 : *cap;
        while (*len + chunk_len + 1 > new_cap) {
            new_cap *= 2;
        }
        *seq = (char *)xrealloc(*seq, (size_t)new_cap);
        *cap = new_cap;
    }

    for (int i = 0; i < chunk_len; ++i) {
        char ch = (char)toupper((unsigned char)chunk[i]);
        if (base_index(ch) < 0) {
            fprintf(stderr, "Invalid DNA base: %c\n", chunk[i]);
            exit(1);
        }
        (*seq)[(*len)++] = ch;
    }
    (*seq)[*len] = '\0';
}

static void trim_newline(char *line) {
    size_t len = strlen(line);
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
        line[--len] = '\0';
    }
}

static StringList read_fasta_or_plain(FILE *fp) {
    StringList reads;
    string_list_init(&reads);

    char line[8192];
    char *current = NULL;
    int current_len = 0;
    int current_cap = 0;
    int saw_header = 0;

    while (fgets(line, sizeof(line), fp) != NULL) {
        trim_newline(line);
        char *text = line;
        while (*text != '\0' && isspace((unsigned char)*text)) {
            ++text;
        }
        if (*text == '\0') {
            continue;
        }

        if (*text == '>') {
            saw_header = 1;
            if (current_len > 0) {
                string_list_push(&reads, current);
                current = NULL;
                current_len = 0;
                current_cap = 0;
            }
            continue;
        }

        if (saw_header) {
            append_sequence_chunk(&current, &current_len, &current_cap, text);
        } else {
            char *seq = NULL;
            int seq_len = 0;
            int seq_cap = 0;
            append_sequence_chunk(&seq, &seq_len, &seq_cap, text);
            string_list_push(&reads, seq);
        }
    }

    if (current_len > 0) {
        string_list_push(&reads, current);
    } else {
        free(current);
    }

    return reads;
}

static int all_reads_same_length(const StringList *reads, int *read_len) {
    if (reads->count == 0) {
        return 0;
    }
    int len = (int)strlen(reads->items[0]);
    for (int i = 1; i < reads->count; ++i) {
        if ((int)strlen(reads->items[i]) != len) {
            return 0;
        }
    }
    *read_len = len;
    return 1;
}

static size_t fft_offset(int read_index, int base, int n_fft) {
    return ((size_t)read_index * BASE_COUNT + (size_t)base) * (size_t)n_fft;
}

static void fill_channel_fft(
    Complex *fft_cache,
    const StringList *reads,
    int read_index,
    int base,
    int n_fft,
    int reversed
) {
    Complex *slot = fft_cache + fft_offset(read_index, base, n_fft);
    const char *seq = reads->items[read_index];
    int len = (int)strlen(seq);

    for (int i = 0; i < n_fft; ++i) {
        slot[i].re = 0.0;
        slot[i].im = 0.0;
    }

    for (int i = 0; i < len; ++i) {
        char ch = reversed ? seq[len - 1 - i] : seq[i];
        slot[i].re = base_index(ch) == base ? 1.0 : 0.0;
    }

    fft(slot, n_fft, 0);
}

static Complex *build_fft_cache(const StringList *reads, int n_fft, int reversed) {
    Complex *cache = (Complex *)xmalloc(
        (size_t)reads->count * BASE_COUNT * (size_t)n_fft * sizeof(Complex)
    );
    for (int read_index = 0; read_index < reads->count; ++read_index) {
        for (int base = 0; base < BASE_COUNT; ++base) {
            fill_channel_fft(cache, reads, read_index, base, n_fft, reversed);
        }
    }
    return cache;
}

static EdgeList find_fft_overlaps_equal_length(
    const StringList *reads,
    int read_len,
    int min_overlap,
    double min_identity
) {
    int n_conv = 2 * read_len - 1;
    int n_fft = next_power_of_two(n_conv);
    Complex *forward_fft = build_fft_cache(reads, n_fft, 0);
    Complex *reverse_fft = build_fft_cache(reads, n_fft, 1);
    Complex *work = (Complex *)xmalloc((size_t)n_fft * sizeof(Complex));
    int *corr = (int *)xcalloc((size_t)n_conv, sizeof(int));

    EdgeList edges;
    edge_list_init(&edges);

    for (int src = 0; src < reads->count; ++src) {
        for (int dst = 0; dst < reads->count; ++dst) {
            if (src == dst) {
                continue;
            }

            memset(corr, 0, (size_t)n_conv * sizeof(int));

            for (int base = 0; base < BASE_COUNT; ++base) {
                Complex *left = forward_fft + fft_offset(src, base, n_fft);
                Complex *right_rev = reverse_fft + fft_offset(dst, base, n_fft);
                for (int k = 0; k < n_fft; ++k) {
                    work[k] = c_mul(left[k], right_rev[k]);
                }

                fft(work, n_fft, 1);
                for (int i = 0; i < n_conv; ++i) {
                    corr[i] += (int)llround(work[i].re);
                }
            }

            int best_overlap = 0;
            int best_matches = 0;
            double best_identity = -1.0;
            int max_overlap = read_len;

            for (int overlap = min_overlap; overlap <= max_overlap; ++overlap) {
                int idx = 2 * read_len - overlap - 1;
                int matches = corr[idx];
                double identity = (double)matches / (double)overlap;
                if (identity < min_identity) {
                    continue;
                }

                if (
                    identity > best_identity ||
                    (identity == best_identity && overlap > best_overlap)
                ) {
                    best_identity = identity;
                    best_overlap = overlap;
                    best_matches = matches;
                }
            }

            if (best_overlap > 0) {
                Edge edge = {
                    src,
                    dst,
                    best_overlap,
                    best_matches,
                    best_identity,
                };
                edge_list_push(&edges, edge);
            }
        }
    }

    free(corr);
    free(work);
    free(reverse_fft);
    free(forward_fft);
    return edges;
}

static int parse_int(const char *text, const char *name) {
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 0 || value > 2147483647L) {
        fprintf(stderr, "Invalid integer for %s: %s\n", name, text);
        exit(2);
    }
    return (int)value;
}

static double parse_double(const char *text, const char *name) {
    char *end = NULL;
    errno = 0;
    double value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0') {
        fprintf(stderr, "Invalid number for %s: %s\n", name, text);
        exit(2);
    }
    return value;
}

static void print_usage(const char *argv0) {
    fprintf(
        stderr,
        "Usage:\n"
        "  %s --benchmark <reads.fa|-> [--min-overlap N] [--min-identity X] [--emit-edges 0|1]\n",
        argv0
    );
}

int main(int argc, char **argv) {
    const char *input_path = NULL;
    int min_overlap = 1;
    double min_identity = 0.8;
    int emit_edges = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--benchmark") == 0) {
            if (i + 1 >= argc) {
                die("--benchmark requires a FASTA path or -");
            }
            input_path = argv[++i];
        } else if (strcmp(argv[i], "--min-overlap") == 0) {
            if (i + 1 >= argc) {
                die("--min-overlap requires a value");
            }
            min_overlap = parse_int(argv[++i], "--min-overlap");
        } else if (strcmp(argv[i], "--min-identity") == 0) {
            if (i + 1 >= argc) {
                die("--min-identity requires a value");
            }
            min_identity = parse_double(argv[++i], "--min-identity");
        } else if (strcmp(argv[i], "--emit-edges") == 0) {
            if (i + 1 >= argc) {
                die("--emit-edges requires 0 or 1");
            }
            emit_edges = parse_int(argv[++i], "--emit-edges") != 0;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 2;
        }
    }

    if (input_path == NULL) {
        print_usage(argv[0]);
        return 2;
    }

    FILE *fp = NULL;
    if (strcmp(input_path, "-") == 0) {
        fp = stdin;
    } else {
        fp = fopen(input_path, "r");
        if (fp == NULL) {
            perror(input_path);
            return 1;
        }
    }

    StringList reads = read_fasta_or_plain(fp);
    if (fp != stdin) {
        fclose(fp);
    }
    if (reads.count == 0) {
        die("no reads found");
    }

    int read_len = 0;
    if (!all_reads_same_length(&reads, &read_len)) {
        die("C FFT prototype currently requires equal-length reads");
    }
    if (min_overlap > read_len) {
        die("min-overlap must not exceed read length");
    }

    double started = monotonic_seconds();
    EdgeList edges = find_fft_overlaps_equal_length(
        &reads,
        read_len,
        min_overlap,
        min_identity
    );
    double seconds = monotonic_seconds() - started;

    printf("method\tc_fft_no_gap\n");
    printf("reads\t%d\n", reads.count);
    printf("read_len\t%d\n", read_len);
    printf("pairs_scanned\t%d\n", reads.count * (reads.count - 1));
    printf("candidates\t%d\n", edges.count);
    printf("min_overlap\t%d\n", min_overlap);
    printf("min_identity\t%.12g\n", min_identity);
    printf("seconds\t%.9f\n", seconds);

    if (emit_edges) {
        printf("edges_begin\n");
        for (int i = 0; i < edges.count; ++i) {
            Edge edge = edges.items[i];
            printf(
                "%d\t%d\t%d\t%d\t%.12g\n",
                edge.src,
                edge.dst,
                edge.overlap,
                edge.matches,
                edge.identity
            );
        }
        printf("edges_end\n");
    }

    free(edges.items);
    string_list_free(&reads);
    return 0;
}
