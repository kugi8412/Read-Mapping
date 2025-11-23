# mapper.py
# -*- coding: utf-8 -*-

import os
import sys
import time
import psutil
import argparse

import numpy as np

from Bio import SeqIO
from array import array
from collections import Counter
from typing import List, Dict, Tuple


INF = 10**9
PARAM_K = 15
PARAM_STEP = 29
PARAM_TOP_N = 5
PARAM_SLACK = 100
PARAM_MAX_ERR = 0.12
PARAM_MAX_HITS = 800


def get_rss_bytes() -> int:
    """ Zwraca aktualne zużycie pamięci RSS w bajtach.
    """
    p = psutil.Process(os.getpid())
    rss = int(p.memory_info().rss)
    return rss


def human_bytes(n: int) -> str:
    """ Konwertuje liczbę bajtów na czytelny
    format (KB, MB, GB, ...).
    """
    if n < 1024:
        return f"{n} B"
    for unit in ("KB", "MB", "GB", "TB"):
        n /= 1024.0
        if n < 1024.0:
            return f"{n:3.2f} {unit}"

    return f"{n:.2f} PB"


def check_memory_and_report(label: str,
                            mem_limit_bytes: int = None,
                            fail_on_exceed: bool = False
                            ) -> bool:
    rss = get_rss_bytes()
    print(f"[MEMORY] {label}: RSS = {human_bytes(rss)} ({rss} bytes)")
    if mem_limit_bytes is not None:
        if rss > mem_limit_bytes:
            print(f"[MEMORY][WARNING] Memory limit exceeded: {human_bytes(rss)} > {human_bytes(mem_limit_bytes)}")
            if fail_on_exceed:
                print("[MEMORY][ERROR] Exiting because --fail-on-memory flag set.")
                sys.exit(2)

            return False

    return True


# DC3 / Karkkainen-Sanders
def radixpass(a: array,
              b: array,
              r: array,
              n: int,
              k: int
              ) -> None:
    c = array("i", [0] * (k + 1))
    for i in range(n):
        c[r[a[i]]] += 1

    sumv = 0
    for i in range(k + 1):
        freq = c[i]
        c[i] = sumv
        sumv += freq

    for i in range(n):
        idx = r[a[i]]
        b[c[idx]] = a[i]
        c[idx] += 1


def direct_kark_sort(s: List[str]) -> array:
    alphabet = [None] + sorted(set(s))
    k = len(alphabet)
    n = len(s)
    t = dict((c, i) for i,c in enumerate(alphabet))
    SA = array('i', [0]*(n+3))
    kark_sort(array('i', [t[c] for c in s] + [0] * 3), SA, n, k)
    return SA[:n]


def kark_sort(s: array,
              suffix_array: array,
              n: int,
              k: int
              ) -> None:
    n0  = (n + 2) // 3
    n1  = (n + 1) // 3
    n2  = n // 3
    n02 = n0 + n2

    S_a12 = array('i', [0] * (n02 + 3))
    S_a0  = array('i', [0] * n0)

    s12 = [i for i in range(n + (n0 - n1)) if i % 3]
    s12.extend([0]*3)
    s12 = array('i', s12)

    radixpass(s12, S_a12, s[2:], n02, k)
    radixpass(S_a12, s12, s[1:], n02, k)
    radixpass(s12, S_a12, s, n02, k)

    name = 0
    c0 = c1 = c2 = -1
    for i in range(n02):
        pos = S_a12[i]
        if s[pos] != c0 or s[pos + 1] != c1 or s[pos + 2] != c2:
            name += 1
            c0 = s[pos]
            c1 = s[pos + 1]
            c2 = s[pos + 2]
        if pos % 3 == 1:
            s12[pos // 3] = name
        else:
            s12[pos // 3 + n0] = name

    if name < n02:
        kark_sort(s12, S_a12, n02, name + 1)
        for i in range(n02):
            s12[S_a12[i]] = i + 1
    else:
        for i in range(n02):
            S_a12[s12[i] - 1] = i

    s0 = array('i', [S_a12[i] * 3 for i in range(n02) if S_a12[i] < n0])
    radixpass(s0, S_a0, s, n0, k)

    p = j = k = 0
    t = n0 - n1
    while k < n:
        if t < n02:
            if S_a12[t] < n0:
                i = S_a12[t] * 3 + 1
            else:
                i = (S_a12[t] - n0) * 3 + 2
        else:
            i = -1
        j = S_a0[p] if p < n0 else -1

        if i == -1:
            suffix_array[k] = j
            p += 1
            k += 1
            continue
        if j == -1:
            suffix_array[k] = i
            t += 1
            k += 1
            continue

        if S_a12[t] < n0:
            a1 = s[i]
            b1 = s[j]
            if a1 != b1:
                take_i = a1 < b1
            else:
                ri = s12[S_a12[t] + n0]
                rj = s12[j // 3]
                take_i = ri <= rj
        else:
            a1 = s[i]
            a2 = s[i+1]
            b1 = s[j]
            b2 = s[j+1]
            if a1 != b1:
                take_i = a1 < b1
            elif a2 != b2:
                take_i = a2 < b2
            else:
                ri = s12[S_a12[t] - n0 + 1]
                rj = s12[j // 3 + n0]
                take_i = ri <= rj

        if take_i:
            suffix_array[k] = i
            t += 1
        else:
            suffix_array[k] = j
            p += 1

        k += 1


def build_SA_from_string(s: List[str]) -> List[int]:
    return list(direct_kark_sort(s))


def read_fasta_concat(path: str) -> str:
    recs = list(SeqIO.parse(path, "fasta"))
    if not recs:
        return ""

    seq = str(recs[0].seq)
    for r in recs[1:]:
        seq += "N" + str(r.seq)

    return seq


def read_reads(path: str) -> List[Tuple[str, str]]:
    recs = list(SeqIO.parse(path, "fasta"))
    return [(r.id, str(r.seq)) for r in recs]  # assume uppercase


def build_fm_index(reference: str) -> Dict:
    if '$' in reference:
        raise ValueError("Reference contains $ character, which is reserved for FM-index.")

    text = reference + "$"
    suffix_array = build_SA_from_string(list(text))
    n = len(text)
    sa = np.array(suffix_array, dtype=np.int32)
    alphabet = sorted(set(text))
    if alphabet[0] != '$':
        alphabet.remove('$')
        alphabet = ['$'] + alphabet

    char_to_idx = {c:i for i,c in enumerate(alphabet)}
    sigma = len(alphabet)
    text_int = np.fromiter((char_to_idx[c] for c in text), dtype=np.int32, count=n)
    sa_minus_1 = sa - 1
    sa_minus_1[sa_minus_1 < 0] = n - 1
    bwt = text_int[sa_minus_1]
    counts = np.bincount(text_int, minlength=sigma)
    Cvals = np.zeros(sigma, dtype=np.int32)
    Cvals[1:] = np.cumsum(counts)[:-1]
    C = {c: int(Cvals[char_to_idx[c]]) for c in alphabet}
    occ = np.zeros((sigma, n + 1), dtype=np.int32)
    for c_idx in range(sigma):
        mask = (bwt == c_idx).astype(np.int32)
        occ[c_idx, 1:] = np.cumsum(mask)

    fm = {
        'text': text,
        'n': n,
        'sa': sa,
        'bwt': bwt,
        'alphabet': alphabet,
        'char_to_idx': char_to_idx,
        'C': C,
        'occ': occ
    }
    return fm


# Backward search for uknown pattern
def bw_backward_search(fm, pattern):
    sp, ep = 0, fm['n']
    occ = fm['occ']
    char_to_idx = fm['char_to_idx']
    Cmap = fm['C']
    for ch in reversed(pattern):
        idx = char_to_idx[ch]
        cval = Cmap[ch]
        sp = cval + int(occ[idx, sp])
        ep = cval + int(occ[idx, ep])
        if sp >= ep:
            return 0, 0

    return sp, ep


# Seeding & candidates
def generate_seeds(read: str,
                   k: int = 17,
                   step: int = 50
                   ) -> List[Tuple[int, str]]:
    L = len(read)
    if L < k:
        return []

    seeds = []
    i = 0
    while i + k <= L:
        seeds.append((i, read[i:i+k]))
        i += step

    if seeds and seeds[-1][0] + k < L:
        pos = max(0, L - k)
        seeds.append((pos, read[pos:pos+k]))

    return seeds


def gather_candidates(fm: Dict,
                      read: str,
                      k: int = 17,
                      step: int = 50,
                      max_pos_per_seed: int = 1000,
                      top_n: int = 6
                      ) -> List[Tuple[int, int, List[int]]]:
    seeds = generate_seeds(read, k=k, step=step)
    if not seeds:
        return []

    offset_counts = Counter()
    occ_lists = {}
    sa = fm['sa']
    for pos, seed in seeds:
        sp, ep = bw_backward_search(fm, seed)
        if sp >= ep:
            continue

        sa_slice = sa[sp:ep]
        if sa_slice.size > max_pos_per_seed:
            sa_slice = sa_slice[:max_pos_per_seed]

        for sa_pos in sa_slice:
            sa_pos = int(sa_pos)
            if sa_pos >= fm['n'] - 1:
                continue

            offset = sa_pos - pos
            offset_counts[offset] += 1
            if offset not in occ_lists:
                occ_lists[offset] = []

            occ_lists[offset].append(sa_pos)
    if not offset_counts:
        return []

    candidates = []
    for off, cnt in offset_counts.most_common(top_n):
        candidates.append((off, cnt, occ_lists.get(off, [])[:200]))

    return candidates

def banded_align(read: str,
                 ref_region: str,
                 region_start: int,
                 max_errors: int
                 ) -> Tuple[int, int, int]:
    m = len(read); n = len(ref_region)
    if n == 0:
        return (m + 1, region_start, region_start - 1)

    band = max(10, max_errors * 2 + 5)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [INF] * (n + 1)
        ch = read[i - 1]
        j0 = max(1, i - band)
        j1 = min(n, i + band)
        curr[0] = i
        prev_loc = prev
        curr_loc = curr
        ref_loc = ref_region
        for j in range(j0, j1 + 1):
            cost_sub = 0 if ch == ref_loc[j - 1] else 1
            v_diag = prev_loc[j - 1] + cost_sub
            v_up = prev_loc[j] + 1
            v_left = curr_loc[j - 1] + 1
            best = v_diag
            if v_up < best:
                best = v_up
            if v_left < best:
                best = v_left

            curr_loc[j] = best
        prev = curr_loc
        if min(prev) > max_errors:
            return (min(prev), region_start, region_start + n - 1)

    best_cost = min(prev)
    best_j = prev.index(best_cost)
    start_j = max(0, best_j - m - 5)
    sub_ref = ref_region[start_j:best_j]
    sub_n = len(sub_ref)

    # full DP on small window to get start
    dp = [[0] * (sub_n + 1) for _ in range(m + 1)]
    tb = [[0] * (sub_n + 1) for _ in range(m + 1)]
    for j in range(1, sub_n + 1):
        dp[0][j] = 0
    for i in range(1, m + 1):
        dp[i][0] = i

    for i in range(1, m + 1):
        ri = read[i - 1]
        row = dp[i]; prev_row = dp[i - 1]
        for j in range(1, sub_n + 1):
            cost_sub = 0 if ri == sub_ref[j - 1] else 1
            v_diag = prev_row[j-1] + cost_sub
            v_up = prev_row[j] + 1
            v_left = row[j-1] + 1
            best = v_diag
            dir = 0
            if v_up < best:
                best = v_up
                dir = 1
            if v_left < best:
                best = v_left
                dir = 2

            row[j] = best
            tb[i][j] = dir
    i = m
    j = sub_n
    while i > 0:
        dir = tb[i][j]
        if dir == 0:
            i -= 1
            j -= 1
        elif dir == 1:
            i -= 1
        else:
            j -= 1

    start_in_sub = j
    start_ref = region_start + start_j + start_in_sub
    end_ref = region_start + best_j - 1
    return (best_cost, start_ref, end_ref)

# Mapping (forward-only, no N-checks)
def map_reads(fm: Dict,
              reads: List[Tuple[str, str]],
              params: Dict
              ) -> List[Tuple[str, int, int]]:
    results = []
    ref_len = fm['n'] - 1
    k = params['k']
    step = params['step']
    slack = params['slack']
    max_seed_hits = params['max_seed_hits']
    error_rate = params['error_rate']
    for rid, seq in reads:
        s = seq
        best_hit = None
        best_cost = INF
        candidates = gather_candidates(fm, s, k=k, step=step, max_pos_per_seed=max_seed_hits, top_n=params['top_n'])
        if not candidates:
            print(f'No candidates for read {rid}')
            candidates = gather_candidates(fm, s, k=k, step=max(5, k // 2), max_pos_per_seed=max_seed_hits, top_n=params['top_n'])
        if not candidates:
            print(f'No candidates (2nd try) for read {rid}, skipping')
            continue

        max_err = max(5, int(error_rate * len(s)))
        for off, _, _ in candidates:
            region_start = max(0, off - slack)
            region_end = min(ref_len, off + len(s) + slack)
            if region_start >= region_end:
                continue
            ref_region = fm['text'][region_start:region_end]
            cost, st, en = banded_align(s, ref_region, region_start, max_err)
            if cost < best_cost:
                best_cost = cost
                best_hit = (rid, st, en, cost)

        if best_hit is not None and best_cost <= int(error_rate * len(s)):
            rid_out = best_hit[0]
            start1 = best_hit[1] + 1
            end1 = best_hit[2] + 1
            results.append((rid_out, start1, end1))

    return results


def main():
    parser = argparse.ArgumentParser(description="FM-index mapper (forward-only, uppercase only) with memory checks.")
    parser.add_argument("reference", help="Reference FASTA")
    parser.add_argument("reads", help="Reads FASTA")
    parser.add_argument("output", help="Output file")
    parser.add_argument("--k", type=int, default=PARAM_K, help="Seed k-mer length")
    parser.add_argument("--step", type=int, default=PARAM_STEP, help="Seed step")
    parser.add_argument("--slack", type=int, default=PARAM_SLACK, help="Slack around candidate (nt)")
    parser.add_argument("--max_seed_hits", type=int, default=PARAM_MAX_HITS, help="Max occurrences per seed considered")
    parser.add_argument("--error_rate", type=float, default=PARAM_MAX_ERR, help="Allowed error rate per read")
    parser.add_argument("--top_n", type=int, default=PARAM_TOP_N, help="Top candidate offsets to try")
    parser.add_argument("--mem-limit", type=str, default="1GB", help="Memory limit to check (e.g. '1GB' or '800MB')")
    parser.add_argument("--fail-on-memory", action='store_true', help="Exit with error if memory limit exceeded")
    args = parser.parse_args()

    # parse memory limit
    def parse_mem(s: str) -> int:
        s = s.strip().upper()
        if s.endswith("GB"):
            return int(float(s[:-2]) * 1024**3)
        if s.endswith("MB"):
            return int(float(s[:-2]) * 1024**2)
        if s.endswith("KB"):
            return int(float(s[:-2]) * 1024)
        return int(s)

    mem_limit_bytes = parse_mem(args.mem_limit)

    t0 = time.time()
    print("Reading reference...")
    ref_seq = read_fasta_concat(args.reference)
    if not ref_seq:
        print("Empty reference!")
        sys.exit(1)

    check_memory_and_report("After reading reference", mem_limit_bytes, args.fail_on_memory)
    print(f"Reference length: {len(ref_seq)}")

    t_build_start = time.time()
    fm = build_fm_index(ref_seq)
    t_build = time.time() - t_build_start

    print(f"FM-index built. n={fm['n']}. Time: {t_build:.1f}s")
    check_memory_and_report("After building FM-index", mem_limit_bytes, args.fail_on_memory)

    reads = read_reads(args.reads)
    print(f"Reads count: {len(reads)}")
    check_memory_and_report("After reading reads", mem_limit_bytes, args.fail_on_memory)

    params = {
        'k': args.k,
        'step': args.step,
        'slack': args.slack,
        'max_seed_hits': args.max_seed_hits,
        'error_rate': args.error_rate,
        'top_n': args.top_n
    }

    t1 = time.time()
    mapped = map_reads(fm, reads, params)
    tmap = time.time() - t1
    print(f"Mapping finished in {tmap:.1f}s; mapped {len(mapped)} reads.")

    check_memory_and_report("after mapping", mem_limit_bytes, args.fail_on_memory)

    with open(args.output, "w") as out:
        for rid, s, e in mapped:
            out.write(f"{rid}\t{s}\t{e}\n")

    print("Total time: {:.2f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()