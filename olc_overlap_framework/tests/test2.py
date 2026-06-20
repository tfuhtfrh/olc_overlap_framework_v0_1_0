import numpy as np
import unittest


BASES = "ACGT"


def reverse_complement(seq: str) -> str:
    table = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(table)[::-1].upper()


def one_hot(seq: str):
    """
    DNA配列を A,C,G,T の4チャンネル信号に変換する。
    
    例:
        seq = "AGCT"
        Aチャンネル: [1,0,0,0]
        Cチャンネル: [0,0,1,0]
        Gチャンネル: [0,1,0,0]
        Tチャンネル: [0,0,0,1]
    """
    seq = seq.upper()
    channels = {}
    for b in BASES:
        channels[b] = np.array([1 if x == b else 0 for x in seq], dtype=float)
    return channels


def fft_convolve(a, b):
    """
    FFTを使って畳み込みを計算する。
    """
    n = len(a) + len(b) - 1
    n_fft = 1 << (n - 1).bit_length()

    fa = np.fft.rfft(a, n_fft)
    fb = np.fft.rfft(b, n_fft)

    c = np.fft.irfft(fa * fb, n_fft)[:n]

    # 数値誤差で 2.999999999 などになるので丸める
    return np.rint(c).astype(int)


def suffix_prefix_overlap_scores(suffix_seq: str, prefix_seq: str):
    """
    suffix_seq の suffix と prefix_seq の prefix の overlap を全長について計算する。

    戻り値:
        [
            {
                "overlap": L,
                "matches": 一致数,
                "identity": 一致率
            },
            ...
        ]

    例:
        suffix_seq = "TTAGCT"
        prefix_seq = "AGCTAA"

        suffix_seq の suffix "AGCT"
        prefix_seq の prefix "AGCT"
        が完全一致するので overlap=4, identity=1.0 になる。
    """
    x = suffix_seq.upper()
    y = prefix_seq.upper()

    n = len(x)
    m = len(y)

    x_ch = one_hot(x)
    y_ch = one_hot(y)

    # 全塩基について相互相関を足す
    # corr[s + m - 1] = shift s での一致数
    corr = np.zeros(n + m - 1, dtype=int)

    for b in BASES:
        corr += fft_convolve(x_ch[b], y_ch[b][::-1])

    results = []

    max_L = min(n, m)

    for L in range(1, max_L + 1):
        # y の prefix 長 L が x の suffix 長 L に重なる
        # y の開始位置 shift s は x の中で n - L
        s = n - L

        idx = s + m - 1
        matches = corr[idx]
        identity = matches / L

        results.append({
            "overlap": L,
            "matches": int(matches),
            "identity": float(identity)
        })

    return results


def best_suffix_prefix_overlap(
    suffix_seq: str,
    prefix_seq: str,
    min_overlap: int = 5,
    min_identity: float = 0.8
):
    """
    suffix_seq の suffix と prefix_seq の prefix の中で、
    条件を満たす最良 overlap を返す。
    """
    scores = suffix_prefix_overlap_scores(suffix_seq, prefix_seq)

    candidates = [
        r for r in scores
        if r["overlap"] >= min_overlap and r["identity"] >= min_identity
    ]

    if not candidates:
        return None

    # まず一致率が高いもの、次に overlap が長いものを優先
    best = max(candidates, key=lambda r: (r["identity"], r["overlap"]))
    return best


def find_best_overlap_with_reverse_complement(
    seq1: str,
    seq2: str,
    min_overlap: int = 5,
    min_identity: float = 0.8
):
    """
    seq1 の suffix と seq2 の prefix を比較する。
    さらに seq2 の reverse-complement も試す。
    """
    seq1 = seq1.upper()
    seq2 = seq2.upper()

    candidates = []

    # seq1 suffix vs seq2 prefix
    best_forward = best_suffix_prefix_overlap(
        seq1, seq2,
        min_overlap=min_overlap,
        min_identity=min_identity
    )

    if best_forward is not None:
        best_forward["orientation"] = "forward"
        best_forward["seq2_used"] = seq2
        candidates.append(best_forward)

    # seq1 suffix vs revcomp(seq2) prefix
    rc_seq2 = reverse_complement(seq2)

    best_rc = best_suffix_prefix_overlap(
        seq1, rc_seq2,
        min_overlap=min_overlap,
        min_identity=min_identity
    )

    if best_rc is not None:
        best_rc["orientation"] = "reverse_complement"
        best_rc["seq2_used"] = rc_seq2
        candidates.append(best_rc)

    if not candidates:
        return None

    best = max(candidates, key=lambda r: (r["identity"], r["overlap"]))
    return best


class FFTOverlapTests(unittest.TestCase):
    LEFT_READ = (
        "CACAGCTTCTGTAGCGAGGTGCGACGTTCTTCAAGTAATCTATTCAGGAT"
        "GAGAAACCACGAAACAGATCTACTTGTACA"
    )
    RIGHT_READ = (
        "GAGAAACCACGAAACAGATCTACTTGTACAGCCTGTGCTCATGGCGTTCT"
        "ACTATTGTTTCCTCTCGCAACAAGCCGCTA"
    )
    RIGHT_READ_TWO_SUBSTITUTIONS = (
        "GAGAAACAACGAAACAGATCTCCTTGTACAGCCTGTGCTCATGGCGTTCT"
        "ACTATTGTTTCCTCTCGCAACAAGCCGCTA"
    )

    def test_exact_suffix_prefix_overlap(self):
        result = best_suffix_prefix_overlap(
            "TTAGCT",
            "AGCTAA",
            min_overlap=4,
            min_identity=1.0,
        )

        self.assertEqual(result, {
            "overlap": 4,
            "matches": 4,
            "identity": 1.0,
        })

    def test_finds_true_overlap_with_two_substitutions_and_no_gaps(self):
        result = best_suffix_prefix_overlap(
            self.LEFT_READ,
            self.RIGHT_READ_TWO_SUBSTITUTIONS,
            min_overlap=25,
            min_identity=0.90,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["overlap"], 30)
        self.assertEqual(result["matches"], 28)
        self.assertAlmostEqual(result["identity"], 28 / 30)

    def test_rejects_same_overlap_when_identity_threshold_is_too_high(self):
        result = best_suffix_prefix_overlap(
            self.LEFT_READ,
            self.RIGHT_READ_TWO_SUBSTITUTIONS,
            min_overlap=30,
            min_identity=0.95,
        )

        self.assertIsNone(result)

    def test_detects_reverse_complement_orientation_with_substitutions(self):
        reverse_oriented_read = reverse_complement(
            self.RIGHT_READ_TWO_SUBSTITUTIONS
        )

        result = find_best_overlap_with_reverse_complement(
            self.LEFT_READ,
            reverse_oriented_read,
            min_overlap=30,
            min_identity=0.90,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["orientation"], "reverse_complement")
        self.assertEqual(result["seq2_used"], self.RIGHT_READ_TWO_SUBSTITUTIONS)
        self.assertEqual(result["overlap"], 30)
        self.assertEqual(result["matches"], 28)

    def test_fft_scores_match_direct_no_gap_counting(self):
        scores = suffix_prefix_overlap_scores(
            self.LEFT_READ,
            self.RIGHT_READ_TWO_SUBSTITUTIONS,
        )

        for score in scores:
            overlap = score["overlap"]
            left_suffix = self.LEFT_READ[-overlap:]
            right_prefix = self.RIGHT_READ_TWO_SUBSTITUTIONS[:overlap]
            expected_matches = sum(
                left_base == right_base
                for left_base, right_base in zip(left_suffix, right_prefix)
            )
            self.assertEqual(score["matches"], expected_matches)
            self.assertAlmostEqual(
                score["identity"],
                expected_matches / overlap,
            )


if __name__ == "__main__":
    unittest.main()
