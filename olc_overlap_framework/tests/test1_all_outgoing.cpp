#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>

using namespace std;

// ============================================================
// 文字コード
// ============================================================

static inline int dna_code(char ch) {
    switch (ch) {
        case 'A': return 1;
        case 'G': return 2;
        case 'T': return 3;
        case 'C': return 4;
        default:
            throw invalid_argument(string("無効文字 '") + ch + "' が含まれている。使用可能文字は A, G, T, C のみである。");
    }
}

// ============================================================
// rolling hash 用定数
// double hash にして衝突確率をかなり下げる
// ============================================================

static constexpr uint64_t MOD1  = 1'000'000'007ULL;
static constexpr uint64_t MOD2  = 1'000'000'009ULL;
static constexpr uint64_t BASE1 = 911'382'323ULL;
static constexpr uint64_t BASE2 = 972'663'749ULL;

static inline uint64_t pack_hash(uint32_t h1, uint32_t h2) {
    return (static_cast<uint64_t>(h1) << 32) | static_cast<uint64_t>(h2);
}

static inline uint64_t pack_pair(int src, int dst) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(src)) << 32)
         | static_cast<uint64_t>(static_cast<uint32_t>(dst));
}

// ============================================================
// 圧縮結果 / overlap edge
// ============================================================

struct CompressionResult {
    vector<string> unique_seqs;                         // class_id -> sequence
    vector<vector<int>> class_members;                  // class_id -> original read ids
    vector<int> read_to_class;                          // original read id -> class_id
    unordered_map<string, int> seq_to_class;             // sequence -> class_id
};

struct OverlapEdge {
    int src_class = -1;
    int dst_class = -1;
    int overlap_len = 0;
};

struct ReadOverlapInfo {
    int src_read = -1;
    int src_class = -1;
    int dst_class = -1;
    int dst_read_rep = -1;
    int overlap_len = 0;
    int src_count = 0;
    int dst_count = 0;
};

// ============================================================
// 入力チェック
// ============================================================

void validate_dna_sequences(const vector<string>& seqs) {
    for (size_t idx = 0; idx < seqs.size(); ++idx) {
        if (seqs[idx].empty()) {
            throw invalid_argument("seqs[" + to_string(idx) + "] が空文字列である。");
        }
        for (char ch : seqs[idx]) {
            (void)dna_code(ch);
        }
    }
}

// ============================================================
// 同一 read 圧縮
// ============================================================

CompressionResult compress_reads(const vector<string>& seqs) {
    validate_dna_sequences(seqs);

    CompressionResult comp;
    comp.seq_to_class.reserve(seqs.size() * 2 + 1);
    comp.unique_seqs.reserve(seqs.size());
    comp.class_members.reserve(seqs.size());
    comp.read_to_class.reserve(seqs.size());

    for (int read_id = 0; read_id < static_cast<int>(seqs.size()); ++read_id) {
        const string& seq = seqs[read_id];
        auto it = comp.seq_to_class.find(seq);

        int class_id;
        if (it == comp.seq_to_class.end()) {
            class_id = static_cast<int>(comp.unique_seqs.size());
            comp.seq_to_class.emplace(seq, class_id);
            comp.unique_seqs.push_back(seq);
            comp.class_members.emplace_back();
        } else {
            class_id = it->second;
        }

        comp.class_members[class_id].push_back(read_id);
        comp.read_to_class.push_back(class_id);
    }

    return comp;
}

// ============================================================
// rolling hash
// ============================================================

struct RollingHashPowers {
    vector<uint32_t> pow1;
    vector<uint32_t> pow2;
};

RollingHashPowers build_powers(int max_len) {
    RollingHashPowers powers;
    powers.pow1.assign(max_len + 1, 1);
    powers.pow2.assign(max_len + 1, 1);

    for (int i = 0; i < max_len; ++i) {
        powers.pow1[i + 1] = static_cast<uint32_t>((static_cast<uint64_t>(powers.pow1[i]) * BASE1) % MOD1);
        powers.pow2[i + 1] = static_cast<uint32_t>((static_cast<uint64_t>(powers.pow2[i]) * BASE2) % MOD2);
    }

    return powers;
}

class DoubleRollingHash {
public:
    explicit DoubleRollingHash(const string& s_, const RollingHashPowers& powers)
        : s(s_), n(static_cast<int>(s_.size())), pow1(&powers.pow1), pow2(&powers.pow2) {
        pref1.assign(n + 1, 0);
        pref2.assign(n + 1, 0);

        for (int i = 0; i < n; ++i) {
            uint64_t x = static_cast<uint64_t>(dna_code(s[i]));
            pref1[i + 1] = static_cast<uint32_t>((static_cast<uint64_t>(pref1[i]) * BASE1 + x) % MOD1);
            pref2[i + 1] = static_cast<uint32_t>((static_cast<uint64_t>(pref2[i]) * BASE2 + x) % MOD2);
        }
    }

    uint64_t substring_hash(int l, int r) const {
        int len = r - l;

        uint64_t a1 = (static_cast<uint64_t>(pref1[l]) * (*pow1)[len]) % MOD1;
        uint64_t a2 = (static_cast<uint64_t>(pref2[l]) * (*pow2)[len]) % MOD2;

        uint32_t h1 = static_cast<uint32_t>((static_cast<uint64_t>(pref1[r]) + MOD1 - a1) % MOD1);
        uint32_t h2 = static_cast<uint32_t>((static_cast<uint64_t>(pref2[r]) + MOD2 - a2) % MOD2);

        return pack_hash(h1, h2);
    }

    uint64_t prefix_hash(int length) const {
        return substring_hash(0, length);
    }

    uint64_t suffix_hash(int length) const {
        return substring_hash(n - length, n);
    }

private:
    const string& s;
    int n;
    const vector<uint32_t>* pow1;
    const vector<uint32_t>* pow2;
    vector<uint32_t> pref1;
    vector<uint32_t> pref2;
};

// ============================================================
// all outgoing overlap
//
// 意味:
//   src_class ごとに「一番長い1本」だけではなく、
//   suffix-prefix overlap を持つ dst_class を全部出す。
//
// 注意:
//   同じ (src, dst) pair が複数の L で一致する場合、
//   overlap 長が最大のものだけを残す。
//   L を大きい方から見るので、最初に見つかった L が最大 overlap。
// ============================================================

vector<OverlapEdge> find_all_outgoing_by_class_rolling_hash(
    const vector<string>& unique_seqs,
    int min_overlap = 1,
    bool allow_self = false,
    bool allow_full_length = true,
    bool verify_exact = true
) {
    vector<OverlapEdge> edges;
    if (unique_seqs.empty()) return edges;

    const int U = static_cast<int>(unique_seqs.size());
    vector<int> lengths(U);
    int max_len = 0;
    for (int i = 0; i < U; ++i) {
        lengths[i] = static_cast<int>(unique_seqs[i].size());
        max_len = max(max_len, lengths[i]);
    }

    RollingHashPowers powers = build_powers(max_len);
    vector<DoubleRollingHash> hashers;
    hashers.reserve(U);
    for (const auto& seq : unique_seqs) {
        hashers.emplace_back(seq, powers);
    }

    unordered_set<uint64_t> seen_pairs;
    seen_pairs.reserve(static_cast<size_t>(U) * 8 + 1);

    // overlap 長を長い方から順に処理する。
    // 同じ pair は最初に見つかったものが最大 overlap なので、それ以降は捨てる。
    for (int L = max_len; L >= min_overlap; --L) {
        unordered_map<uint64_t, vector<int>> prefix_map;
        unordered_map<uint64_t, vector<int>> suffix_map;
        prefix_map.reserve(static_cast<size_t>(U) * 2 + 1);
        suffix_map.reserve(static_cast<size_t>(U) * 2 + 1);

        for (int class_id = 0; class_id < U; ++class_id) {
            int n = lengths[class_id];
            if (n < L) continue;

            if (allow_full_length || n > L) {
                prefix_map[hashers[class_id].prefix_hash(L)].push_back(class_id);
                suffix_map[hashers[class_id].suffix_hash(L)].push_back(class_id);
            }
        }

        if (prefix_map.empty() || suffix_map.empty()) continue;

        for (const auto& kv : suffix_map) {
            uint64_t h = kv.first;
            const vector<int>& src_classes = kv.second;

            auto it = prefix_map.find(h);
            if (it == prefix_map.end()) continue;
            const vector<int>& dst_classes = it->second;

            for (int src : src_classes) {
                const string& src_seq = unique_seqs[src];

                for (int dst : dst_classes) {
                    if (!allow_self && dst == src) continue;

                    uint64_t pair_key = pack_pair(src, dst);
                    if (seen_pairs.find(pair_key) != seen_pairs.end()) continue;

                    if (verify_exact) {
                        const string& dst_seq = unique_seqs[dst];
                        if (src_seq.compare(src_seq.size() - static_cast<size_t>(L), static_cast<size_t>(L),
                                            dst_seq, 0, static_cast<size_t>(L)) != 0) {
                            continue;
                        }
                    }

                    seen_pairs.insert(pair_key);
                    edges.push_back(OverlapEdge{src, dst, L});
                }
            }
        }
    }

    sort(edges.begin(), edges.end(), [](const OverlapEdge& a, const OverlapEdge& b) {
        if (a.src_class != b.src_class) return a.src_class < b.src_class;
        if (a.overlap_len != b.overlap_len) return a.overlap_len > b.overlap_len;
        return a.dst_class < b.dst_class;
    });

    return edges;
}

// ============================================================
// class 結果を元 read に戻す
// ============================================================

vector<ReadOverlapInfo> expand_all_outgoing_to_reads(
    const CompressionResult& comp,
    const vector<OverlapEdge>& edges_by_class
) {
    vector<ReadOverlapInfo> result;

    size_t reserve_size = 0;
    for (const auto& e : edges_by_class) {
        reserve_size += comp.class_members[e.src_class].size();
    }
    result.reserve(reserve_size);

    for (const auto& e : edges_by_class) {
        int src_class = e.src_class;
        int dst_class = e.dst_class;
        int overlap_len = e.overlap_len;
        int dst_rep = comp.class_members[dst_class][0];

        for (int read_id : comp.class_members[src_class]) {
            result.push_back(ReadOverlapInfo{
                read_id,
                src_class,
                dst_class,
                dst_rep,
                overlap_len,
                static_cast<int>(comp.class_members[src_class].size()),
                static_cast<int>(comp.class_members[dst_class].size())
            });
        }
    }

    sort(result.begin(), result.end(), [](const ReadOverlapInfo& a, const ReadOverlapInfo& b) {
        if (a.src_read != b.src_read) return a.src_read < b.src_read;
        if (a.overlap_len != b.overlap_len) return a.overlap_len > b.overlap_len;
        return a.dst_class < b.dst_class;
    });

    return result;
}

// ============================================================
// マージ
// ============================================================

string merge_two_sequences(const string& seq1, const string& seq2, int overlap_len) {
    if (overlap_len < 0) {
        throw invalid_argument("overlap_len は 0 以上でなければならない。");
    }
    if (overlap_len > static_cast<int>(seq1.size()) || overlap_len > static_cast<int>(seq2.size())) {
        throw invalid_argument("overlap_len が配列長を超えている。");
    }
    if (overlap_len > 0) {
        if (seq1.compare(seq1.size() - static_cast<size_t>(overlap_len), static_cast<size_t>(overlap_len),
                         seq2, 0, static_cast<size_t>(overlap_len)) != 0) {
            throw invalid_argument("指定 overlap_len では実際には一致していない。");
        }
    }

    return seq1 + seq2.substr(static_cast<size_t>(overlap_len));
}

// ============================================================
// まとめて実行
// ============================================================

struct SolveResult {
    CompressionResult comp;
    vector<OverlapEdge> all_out_by_class;
    vector<ReadOverlapInfo> all_out_by_read;
};

SolveResult solve_all_outgoing_overlap_rolling_hash(
    const vector<string>& seqs,
    int min_overlap = 1,
    bool allow_self = false,
    bool allow_full_length = true,
    bool verify_exact = true
) {
    SolveResult res;
    res.comp = compress_reads(seqs);
    res.all_out_by_class = find_all_outgoing_by_class_rolling_hash(
        res.comp.unique_seqs,
        min_overlap,
        allow_self,
        allow_full_length,
        verify_exact
    );
    res.all_out_by_read = expand_all_outgoing_to_reads(res.comp, res.all_out_by_class);
    return res;
}

// ============================================================
// 表示用
// ============================================================

void print_compression_result(const CompressionResult& comp) {
    cout << "=== read 圧縮結果 ===\n";
    cout << "元 read 数: " << comp.read_to_class.size() << "\n";
    cout << "異なる class 数: " << comp.unique_seqs.size() << "\n\n";

    for (int class_id = 0; class_id < static_cast<int>(comp.unique_seqs.size()); ++class_id) {
        const auto& seq = comp.unique_seqs[class_id];
        const auto& members = comp.class_members[class_id];

        cout << "class=" << class_id
             << ", seq='" << seq << "'"
             << ", count=" << members.size()
             << ", members=[";
        for (size_t i = 0; i < members.size(); ++i) {
            if (i) cout << ", ";
            cout << members[i];
        }
        cout << "]\n";
    }
}

void print_all_out_by_class(
    const CompressionResult& comp,
    const vector<OverlapEdge>& all_out_by_class
) {
    cout << "=== class ごとの all outgoing overlaps ===\n";
    if (all_out_by_class.empty()) {
        cout << "(なし)\n";
        return;
    }

    for (const auto& e : all_out_by_class) {
        int src_class = e.src_class;
        int dst_class = e.dst_class;
        int overlap_len = e.overlap_len;
        const string& src_seq = comp.unique_seqs[src_class];
        const string& dst_seq = comp.unique_seqs[dst_class];
        string merged = merge_two_sequences(src_seq, dst_seq, overlap_len);

        cout << "class " << src_class << " -> class " << dst_class
             << ", overlap=" << overlap_len
             << ", suffix='" << src_seq.substr(src_seq.size() - static_cast<size_t>(overlap_len)) << "'"
             << ", prefix='" << dst_seq.substr(0, static_cast<size_t>(overlap_len)) << "'"
             << ", merged='" << merged << "'\n";
    }
}

void print_all_out_by_read(
    const vector<string>& seqs,
    const CompressionResult& comp,
    const vector<ReadOverlapInfo>& all_out_by_read
) {
    cout << "=== 元 read ごとの all outgoing overlaps ===\n";
    if (all_out_by_read.empty()) {
        cout << "(なし)\n";
        return;
    }

    size_t pos = 0;
    for (int read_id = 0; read_id < static_cast<int>(seqs.size()); ++read_id) {
        bool printed_header = false;
        while (pos < all_out_by_read.size() && all_out_by_read[pos].src_read == read_id) {
            if (!printed_header) {
                cout << "read " << read_id << ":\n";
                printed_header = true;
            }

            const ReadOverlapInfo& info = all_out_by_read[pos];
            int src_class = info.src_class;
            int dst_class = info.dst_class;
            int overlap_len = info.overlap_len;
            int dst_rep = info.dst_read_rep;

            const string& src_seq = comp.unique_seqs[src_class];
            const string& dst_seq = comp.unique_seqs[dst_class];

            cout << "  -> class " << dst_class << " [rep read " << dst_rep << "]"
                 << ", overlap=" << overlap_len
                 << ", suffix='" << src_seq.substr(src_seq.size() - static_cast<size_t>(overlap_len)) << "'"
                 << ", prefix='" << dst_seq.substr(0, static_cast<size_t>(overlap_len)) << "'\n";
            ++pos;
        }

        if (!printed_header) {
            cout << "read " << read_id << ": outgoing なし\n";
        }
    }
}

// ============================================================
// デモ
// ============================================================

void demo() {
    vector<string> seqs = {
        "AGTCATT",   // read 0
        "ATTGCAA",   // read 1
        "CATTGGA",   // read 2
        "TTACCCC",   // read 3
        "GCAATTT",   // read 4
        "AGTCATT",   // read 5, read 0 と同一
        "ATTGCAA"    // read 6, read 1 と同一
    };

    SolveResult res = solve_all_outgoing_overlap_rolling_hash(
        seqs,
        1,      // min_overlap: overlap=1 も出す
        false,  // allow_self
        true,   // allow_full_length
        true    // verify_exact
    );

    cout << "入力 read 一覧:\n";
    for (int i = 0; i < static_cast<int>(seqs.size()); ++i) {
        cout << "  read " << i << ": " << seqs[i] << "\n";
    }
    cout << "\n";

    print_compression_result(res.comp);
    cout << "\n";
    print_all_out_by_class(res.comp, res.all_out_by_class);
    cout << "\n";
    print_all_out_by_read(seqs, res.comp, res.all_out_by_read);
}

string trim_ascii(const string& s) {
    size_t first = 0;
    while (first < s.size() && (s[first] == ' ' || s[first] == '\t' || s[first] == '\r' || s[first] == '\n')) {
        ++first;
    }

    size_t last = s.size();
    while (last > first && (s[last - 1] == ' ' || s[last - 1] == '\t' || s[last - 1] == '\r' || s[last - 1] == '\n')) {
        --last;
    }

    return s.substr(first, last - first);
}

vector<string> read_sequences_from_stream(istream& in) {
    vector<string> seqs;
    string current;
    string line;
    bool saw_fasta_header = false;

    while (getline(in, line)) {
        line = trim_ascii(line);
        if (line.empty()) continue;

        if (line[0] == '>') {
            saw_fasta_header = true;
            if (!current.empty()) {
                seqs.push_back(current);
                current.clear();
            }
            continue;
        }

        if (saw_fasta_header) {
            current += line;
        } else {
            seqs.push_back(line);
        }
    }

    if (!current.empty()) {
        seqs.push_back(current);
    }

    return seqs;
}

void print_usage(const char* argv0) {
    cerr << "Usage:\n"
         << "  " << argv0 << "\n"
         << "  " << argv0 << " --benchmark <reads.fa|-> [--min-overlap N] [--allow-full-length 0|1] [--verify-exact 0|1] [--emit-edges 0|1]\n";
}

int parse_int_arg(const string& value, const string& name) {
    try {
        size_t pos = 0;
        int parsed = stoi(value, &pos);
        if (pos != value.size()) {
            throw invalid_argument("trailing characters");
        }
        return parsed;
    } catch (const exception&) {
        throw invalid_argument("Invalid integer for " + name + ": " + value);
    }
}

int run_benchmark(int argc, char** argv) {
    string input_path;
    int min_overlap = 1;
    bool allow_full_length = true;
    bool verify_exact = true;
    bool emit_edges = false;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--benchmark") {
            if (i + 1 >= argc) {
                throw invalid_argument("--benchmark requires a FASTA path or -");
            }
            input_path = argv[++i];
        } else if (arg == "--min-overlap") {
            if (i + 1 >= argc) {
                throw invalid_argument("--min-overlap requires a value");
            }
            min_overlap = parse_int_arg(argv[++i], "--min-overlap");
        } else if (arg == "--allow-full-length") {
            if (i + 1 >= argc) {
                throw invalid_argument("--allow-full-length requires 0 or 1");
            }
            allow_full_length = parse_int_arg(argv[++i], "--allow-full-length") != 0;
        } else if (arg == "--verify-exact") {
            if (i + 1 >= argc) {
                throw invalid_argument("--verify-exact requires 0 or 1");
            }
            verify_exact = parse_int_arg(argv[++i], "--verify-exact") != 0;
        } else if (arg == "--emit-edges") {
            if (i + 1 >= argc) {
                throw invalid_argument("--emit-edges requires 0 or 1");
            }
            emit_edges = parse_int_arg(argv[++i], "--emit-edges") != 0;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            throw invalid_argument("Unknown argument: " + arg);
        }
    }

    if (input_path.empty()) {
        print_usage(argv[0]);
        return 2;
    }

    vector<string> seqs;
    if (input_path == "-") {
        seqs = read_sequences_from_stream(cin);
    } else {
        ifstream input(input_path);
        if (!input) {
            throw runtime_error("Cannot open input file: " + input_path);
        }
        seqs = read_sequences_from_stream(input);
    }

    auto started = chrono::steady_clock::now();
    SolveResult res = solve_all_outgoing_overlap_rolling_hash(
        seqs,
        min_overlap,
        false,
        allow_full_length,
        verify_exact
    );
    auto finished = chrono::steady_clock::now();
    chrono::duration<double> elapsed = finished - started;

    cout << "method\tcpp_original_all\n"
         << "reads\t" << seqs.size() << "\n"
         << "unique_classes\t" << res.comp.unique_seqs.size() << "\n"
         << "candidates\t" << res.all_out_by_read.size() << "\n"
         << "min_overlap\t" << min_overlap << "\n"
         << "seconds\t" << elapsed.count() << "\n";

    if (emit_edges) {
        cout << "edges_begin\n";
        for (const auto& info : res.all_out_by_read) {
            cout << info.src_read << "\t"
                 << info.dst_read_rep << "\t"
                 << info.overlap_len << "\n";
        }
        cout << "edges_end\n";
    }

    return 0;
}

int main(int argc, char** argv) {
    try {
        if (argc == 1) {
            demo();
            return 0;
        }
        return run_benchmark(argc, argv);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
