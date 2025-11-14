#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from array import array
from collections import Counter


def radixpass(a, b, r, n, k) :
  c = array("i", [0]*(k+1))
  for i in range(n) :
    c[r[a[i]]]+=1

  somme = 0
  for i in range(k+1):
    freq, c[i] = c[i], somme
    somme += freq

  for i in range(n) :
    b[c[r[a[i]]]] = a[i]
    c[r[a[i]]] += 1


def direct_kark_sort(s) :
  alphabet = [None] + sorted(set(s))
  k = len(alphabet)
  n = len(s)
  t = dict((c, i) for i,c in enumerate(alphabet))
  SA = array('i', [0]*(n+3))
  kark_sort(array('i', [t[c] for c in s]+[0]*3), SA, n, k)
  return SA[:n]


def kark_sort(s, SA, n, K):
  n0  = (n+2) // 3
  n1  = (n+1) // 3
  n2  = n // 3
  n02 = n0 + n2
      
  SA12 = array('i', [0]*(n02+3))
  SA0  = array('i', [0]*n0)

  s12 = [i for i in range(n+(n0-n1)) if i%3] 
  s12.extend([0]*3)
  s12 = array('i', s12)

  radixpass(s12, SA12, s[2:], n02, K)
  radixpass(SA12, s12, s[1:], n02, K)
  radixpass(s12, SA12, s, n02, K)

  name = 0
  c0, c1, c2 = -1, -1, -1
  for i in range(n02) :
    if s[SA12[i]] != c0 or s[SA12[i]+1] != c1 or s[SA12[i]+2] != c2 :
      name += 1
      c0 = s[SA12[i]]
      c1 = s[SA12[i]+1]
      c2 = s[SA12[i]+2]
    if SA12[i] % 3 == 1 :
      s12[SA12[i]//3] = name
    else :
      s12[SA12[i]//3 + n0] = name

  if name < n02 :
    kark_sort(s12, SA12, n02, name+1)
    for i in range(n02) :
      s12[SA12[i]] = i+1
  else :
    for i in range(n02) :
      SA12[s12[i]-1] = i

  s0 = array('i',[SA12[i]*3 for i in range(n02) if SA12[i]<n0])
  radixpass(s0, SA0, s, n0, K)
  
  p = j = k = 0
  t = n0 - n1
  while k < n :
    i = SA12[t]*3+1 if SA12[t]<n0 else (SA12[t] - n0)*3 + 2
    j = SA0[p] if p < n0 else 0

    if SA12[t] < n0 :
      test = (s12[SA12[t]+n0] <= s12[j//3]) if(s[i]==s[j]) else (s[i] < s[j])
    elif(s[i]==s[j]) :
      test = s12[SA12[t]-n0+1] <= s12[j//3 + n0] if(s[i+1]==s[j+1]) else s[i+1] < s[j+1]
    else :
      test = s[i] < s[j]

    if(test) :
      SA[k] = i
      t += 1
      if t == n02 :
        k += 1
        while p < n0 :
          SA[k] = SA0[p]
          p += 1
          k += 1
        
    else : 
      SA[k] = j
      p += 1
      if p == n0 :
        k += 1
        while t < n02 :
          SA[k] = (SA12[t] * 3) + 1 if SA12[t] < n0 else ((SA12[t] - n0) * 3) + 2
          t += 1
          k += 1
    k += 1


def build_SA_from_string(s):
    return list(direct_kark_sort(s))


# FASTA reading utility


def read_fasta(path):
    seqs = []
    with open(path) as f:
        name = None
        buf = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seqs.append((name, "".join(buf)))
                name = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if name is not None:
            seqs.append((name, "".join(buf)))
    return seqs


# Reverse complement utility
RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")
def revcomp(s):
    return s.translate(RC_TABLE)[::-1]

# FM index construction

def build_BWT_and_FM(ref):
    text = ref + "$"
    SA = build_SA_from_string(list(text))
    n = len(text)
    BWT = [''] * n
    for i, sa in enumerate(SA):
        if sa == 0:
            BWT[i] = '$'
        else:
            BWT[i] = text[sa - 1]

    # alphabet: unique sorted chars
    alphabet = sorted(set(BWT))
    # cmap = {c:i for i,c in enumerate(alphabet)}
    counts = {c:0 for c in alphabet}
    for ch in text:
        counts[ch] = counts.get(ch,0)+1

    C = {}
    total = 0
    for c in sorted(alphabet):
        C[c] = total
        total += counts.get(c,0)

    occ = {}
    for c in alphabet:
        occ[c] = array('I', [0])

    cur = {c:0 for c in alphabet}
    for ch in BWT:
        for c in alphabet:
            occ[c].append(cur[c])
        cur[ch] += 1

    occ = {}
    cur = {c:0 for c in alphabet}
    occ_len = n + 1
    for c in alphabet:
        occ[c] = array('I', [0])

    for i in range(n):
        ch = BWT[i]
        cur[ch] += 1
        for c in alphabet:
            occ[c].append(cur[c])

    return {
        'text': text,
        'SA': SA,
        'BWT': BWT,
        'alphabet': alphabet,
        'C': C,
        'occ': occ
    }

def bw_backward_search(fm, pattern):
    C = fm['C']
    occ = fm['occ']
    alphabet = fm['alphabet']
    l = 0
    r = len(fm['BWT'])
    for ch in reversed(pattern):
        if ch not in alphabet:
            return (0,0)
        l = C[ch] + occ[ch][l]
        r = C[ch] + occ[ch][r]
        if l >= r:
            return (0,0)
    return (l, r)


# Seed selection, chaining and candidate generation


def generate_seeds(read, k=17, step=50):
    seeds = []
    L = len(read)
    if L < k:
        return seeds

    i = 0
    while i + k <= L:
        seeds.append((i, read[i:i+k]))
        i += step

    if seeds and seeds[-1][0] + k < L:
        pos = max(0, L - k)
        seeds.append((pos, read[pos:pos+k]))
    return seeds

def gather_candidates(fm, read, k=17, step=50, max_pos_per_seed=500):
    seeds = generate_seeds(read, k, step)
    offset_counts = Counter()
    occurrences = {}
    for pos, seed in seeds:
        l,r = bw_backward_search(fm, seed)
        if l>=r:
            continue

        occs = fm['SA'][l:r]
        if len(occs) > max_pos_per_seed:
            occs = occs[:max_pos_per_seed]

        for sa_pos in occs:
            if sa_pos >= len(fm['text']) - 1:
                continue
            offset = sa_pos - pos
            offset_counts[offset] += 1
            occurrences.setdefault(offset, []).append(sa_pos)

    if not offset_counts:
        return []

    min_count = 2
    candidates = []
    for off, cnt in offset_counts.most_common(10):
        if cnt >= min_count or len(candidates) < 3:
            candidates.append((off, cnt, occurrences.get(off,[])))

    return candidates

# Edit-distance DP aligning read to any substring of region
# DP initialization: DP[0][j] = 0 to allow matching read to any substring (free prefix in text)


def align_read_to_region(read, ref_region, region_start, max_allowed_errors):
    m = len(read)
    n = len(ref_region)

    if n == 0:
        return (m, region_start, region_start-1)  # no match

    INF = 10**9
    prev = [0] * (n+1)  # DP[0][j] = 0 (free prefix in ref)
    tb = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        curr = [INF] * (n+1)
        curr[0] = i
        tb[i][0] = 1
        ri = read[i-1]
        # full scan over n (n - read_len + slack, modest)
        for j in range(1, n+1):
            cost_sub = 0 if ri == ref_region[j-1] else 1
            # diag
            v_diag = prev[j-1] + cost_sub
            # up: insertion in ref (i-1 -> i)
            v_up = prev[j] + 1
            # left: deletion in ref
            v_left = curr[j-1] + 1
            best = v_diag
            tb_dir = 0
            if v_up < best:
                best = v_up; tb_dir = 1
            if v_left < best:
                best = v_left; tb_dir = 2
            curr[j] = best
            tb[i][j] = tb_dir
        prev = curr
        # optional pruning
        if min(prev) > max_allowed_errors:
            return (min(prev), region_start, region_start + n - 1)

    # final row
    best_cost = min(prev)
    best_j = prev.index(best_cost)
    # traceback
    i = m
    j = best_j
    while i > 0:
        dir = tb[i][j]
        if dir == 0:
            i -= 1
            j -= 1
        elif dir == 1:
            i -= 1
        elif dir == 2:
            j -= 1

    start_j = j
    start_ref = region_start + start_j
    end_ref = region_start + best_j - 1
    return (best_cost, start_ref, end_ref)

# MAPPING FUNCTION

def map_reads(fm, reads, k=17, step=50, slack=200, max_pos_per_seed=500, error_rate=0.20):
    results = []
    ref_len = len(fm['text']) - 1
    for rid, read in reads:
        read = read.upper()
        best_hit = None
        best_cost = 10**9

        for strand in (0,1):
            seq = read if strand==0 else revcomp(read)
            candidates = gather_candidates(fm, seq, k=k, step=step, max_pos_per_seed=max_pos_per_seed)
            # If no candidates, as fallback try seeds every k/2
            if not candidates:
                candidates = gather_candidates(fm, seq, k=k, step=max(5,k//2), max_pos_per_seed=max_pos_per_seed)

            # Evaluate candidates
            max_err = int(error_rate * len(seq))
            for off, cnt, occs in candidates:
                # Define region to align
                region_start = max(0, off - slack)
                region_end = min(ref_len, off + len(seq) + slack)
                if region_start >= region_end:
                    continue

                ref_region = fm['text'][region_start:region_end]
                cost, s_ref, e_ref = align_read_to_region(seq, ref_region, region_start, max_allowed_errors=max_err)
                if cost < best_cost:
                    best_cost = cost
                    best_hit = (rid, s_ref, e_ref, strand, cost)

        if best_hit is not None and best_cost <= int(0.20 * len(read)):
            rid_out = best_hit[0]
            start1 = best_hit[1] + 1
            end1 = best_hit[2] + 1
            results.append((rid_out, start1, end1))

    return results


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 mapper.py reference.fasta reads.fasta output.txt", file=sys.stderr)
        sys.exit(1)
    ref_fa = sys.argv[1]
    reads_fa = sys.argv[2]
    out_path = sys.argv[3]

    print("Reading reference.", file=sys.stderr)
    refs = read_fasta(ref_fa)
    if not refs:
        print("No reference sequences found", file=sys.stderr)
        sys.exit(1)

    # For simplicity, concatenate all reference sequences with 'N' separators
    ref = refs[0][1].upper()
    if len(refs) > 1:
        for name, seq in refs[1:]:
            ref += 'N' + seq.upper()

    print("Building FM-index.", file=sys.stderr)
    fm = build_BWT_and_FM(ref)
    print("FM-index built. ref length:", len(ref), file=sys.stderr)

    print("Reading reads.", file=sys.stderr)
    reads = read_fasta(reads_fa)

    print("Mapping reads.", file=sys.stderr)
    mapped = map_reads(fm, reads, k=17, step=50, slack=200, max_pos_per_seed=500, error_rate=0.20)

    print("Writing output.", file=sys.stderr)
    with open(out_path, 'w') as out:
        for rid, s, e in mapped:
            out.write(f"{rid}\t{s}\t{e}\n")

    print("Done. Mapped:", len(mapped), "reads", file=sys.stderr)

if __name__ == "__main__":
    main()
    #  python .\FM_index.py .\example_files\reference.fasta .\example_files\reads0.fasta fm_output1.txt
