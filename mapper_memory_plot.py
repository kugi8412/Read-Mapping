# mapper_memory_plot.py
# -*- coding: utf-8 -*-

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import time
import psutil
import argparse
import threading
import numpy as np
import matplotlib.pyplot as plt

from Bio import SeqIO
from collections import Counter
from typing import Dict, Tuple, List

# Config
INF = 10**9
PARAM_K = 15
PARAM_STEP = 29
PARAM_TOP_N = 5
PARAM_SLACK = 100
PARAM_MAX_ERR = 0.12
PARAM_MAX_HITS = 800
CP_INTERVAL = 128


# Memory Monitor
class MemoryMonitor(threading.Thread):
    def __init__(self, interval=0.05):
        super().__init__()
        self.interval = interval
        self.stop_event = threading.Event()
        self.history = []
        self.start_time = time.time()
        self.peak = 0
        self.daemon = True

    def run(self):
        p = psutil.Process(os.getpid())
        while not self.stop_event.is_set():
            try:
                rss = p.memory_info().rss
                if rss > self.peak:
                    self.peak = rss
                self.history.append((time.time() - self.start_time, rss))
            except:
                break
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()

    def plot(self, path="Memory_usage.png"):
        if not self.history:
            return
        t, m = zip(*self.history)
        m_mb = [x / (1024 ** 2) for x in m]
        plt.figure(figsize=(10, 4))
        plt.plot(t, m_mb, label='RSS Usage', color='darkblue')
        plt.fill_between(t, m_mb, color='darkblue', alpha=0.1)
        plt.xlabel("Time [s]")
        plt.ylabel("RSS [MB]")
        plt.title(f"Memory Usage (Peak: {self.peak / 1024**2:.2f} MB)")
        plt.grid(True, linestyle='--')
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"[MONITOR] Plot saved to {path}")


# Suffix Array
def build_sa_numpy(text_bytes: bytes) -> np.ndarray:
    n = len(text_bytes)

    # Convert string to numpy array of uint8
    s = np.frombuffer(text_bytes, dtype=np.uint8)
    rank = s.astype(np.int32)

    # Start SA (0, 1, 2, ... N-1)
    sa = np.arange(n, dtype=np.int32)
    k = 1
    while k < n:
        # Shifted rank
        rank_shifted = np.full_like(rank, -1)
        if k < n:
            rank_shifted[:-k] = rank[k:]
        
        # Stable lexicographical sort by (rank, rank_shifted)
        sa = np.lexsort((rank_shifted, rank))
        old_rank = rank[sa]
        old_rank_shifted = rank_shifted[sa]
        diff = (old_rank[1:] != old_rank[:-1]) | \
               (old_rank_shifted[1:] != old_rank_shifted[:-1])
        new_rank_val = np.empty(n, dtype=np.int32)
        new_rank_val[0] = 0
        np.cumsum(diff, out=new_rank_val[1:])
        rank_new = np.empty_like(rank)
        rank_new[sa] = new_rank_val

        del rank, rank_shifted, old_rank, old_rank_shifted
        gc.collect()
        rank = rank_new

        # Early stopping if all ranks are unique
        if rank[sa[-1]] == n - 1:
            break

        k <<= 1
  
    return sa


# FM Index Construction
def build_fm_index(reference: str) -> Dict:
    if '$' in reference:
        raise ValueError("Reference contains $ character, which is reserved for FM-index.")

    text = reference + "$"
    text_bytes = text.encode('ascii')
    del text
    gc.collect()
    
    n = len(text_bytes)
    print("  -> Building Suffix Array...")
    sa = build_sa_numpy(text_bytes)
    alphabet = sorted(list(set(text_bytes)))
    char_to_idx = {c: i for i, c in enumerate(alphabet)}
    sigma = len(alphabet)
    trans_table = np.zeros(256, dtype=np.uint8)
    for char_code, idx in char_to_idx.items():
        trans_table[char_code] = idx

    print("  -> Converting text...")
    raw_np = np.frombuffer(text_bytes, dtype=np.uint8)
    text_int = trans_table[raw_np]
    sa_minus_1 = sa - 1
    sa_minus_1[sa_minus_1 < 0] = n - 1
    bwt = text_int[sa_minus_1]
    counts = np.bincount(text_int, minlength=sigma)
    Cvals = np.zeros(sigma, dtype=np.int32)
    Cvals[1:] = np.cumsum(counts)[:-1]
    C = {c: int(Cvals[char_to_idx[c]]) for c in alphabet}
    
    print("  -> Building Checkpoints...")
    num_cp = (n + CP_INTERVAL) // CP_INTERVAL + 1
    occ_cp = np.zeros((sigma, num_cp), dtype=np.int32)
    running = np.zeros(sigma, dtype=np.int32)
    
    for i in range(0, n, CP_INTERVAL):
        occ_cp[:, i // CP_INTERVAL] = running
        end_idx = min(i + CP_INTERVAL, n)
        running += np.bincount(bwt[i:end_idx], minlength=sigma)
        
    if n % CP_INTERVAL == 0:
        occ_cp[:, n // CP_INTERVAL] = running
    
    del raw_np, text_int
    gc.collect()
    
    return {
        'text_bytes': text_bytes,
        'n': n,
        'sa': sa,
        'bwt': bwt,
        'alphabet': alphabet,
        'char_to_idx': char_to_idx,
        'C': C,
        'occ_cp': occ_cp,
        'cp_interval': CP_INTERVAL
    }


def get_occ(fm, char_idx: int, i: int) -> int:
    if i == 0: return 0
    cp = i // fm['cp_interval']
    if cp >= fm['occ_cp'].shape[1]: 
        cp = fm['occ_cp'].shape[1] - 1
        
    base = fm['occ_cp'][char_idx, cp]
    start = cp * fm['cp_interval']
    
    if start == i: return int(base)

    # Slice for uint8
    return int(base + np.count_nonzero(fm['bwt'][start:i] == char_idx))


def bw_backward_search(fm: Dict,
                       pattern: bytes
                       ) -> Tuple[int, int]:
    sp, ep = 0, fm['n']
    for c in reversed(pattern):
        if c not in fm['char_to_idx']:
            return 0, 0

        idx = fm['char_to_idx'][c]
        cval = fm['C'][c]
        sp = cval + get_occ(fm, idx, sp)
        ep = cval + get_occ(fm, idx, ep)
        if sp >= ep:
            return 0, 0

    return sp, ep

def generate_seeds(read: str,
                   k: int,
                   step: int
                   ) -> List[Tuple[int, str]]:
    L = len(read)
    if L < k: return []

    seeds = []
    i = 0
    while i + k <= L:
        seeds.append((i, read[i:i+k]))
        i += step

    if seeds and seeds[-1][0] + k < L:
        pos = max(0, L - k)
        seeds.append((pos, read[pos:pos+k]))

    return seeds

def gather_candidates(fm,
                      read,
                      k,
                      step,
                      max_pos_per_seed,
                      top_n
                      ) -> List[Tuple[int, int, List[int]]]:
    seeds = generate_seeds(read, k=k, step=step)
    if not seeds:
        return []

    offset_counts = Counter()
    sa = fm['sa']
    for pos, seed in seeds:
        sp, ep = bw_backward_search(fm, seed.encode('ascii'))
        if sp >= ep: continue

        if ep - sp > max_pos_per_seed:
            ep = sp + max_pos_per_seed

        sa_slice = sa[sp:ep]
        for sa_pos in sa_slice:
            sa_pos = int(sa_pos)
            if sa_pos >= fm['n'] - 1: continue

            offset = sa_pos - pos
            offset_counts[offset] += 1
            
    if not offset_counts:
        return []

    candidates = []
    for off, cnt in offset_counts.most_common(top_n):
        candidates.append((off, cnt, [])) 

    return candidates


def banded_align(read: str,
                 ref_bytes: bytes,
                 region_start: int,
                 max_err: int
                 ) -> Tuple[int, int, int]:
    m = len(read)
    n = len(ref_bytes)
    if n == 0:
        return INF, region_start, region_start

    read_bytes = read.encode("ascii")
    band = max(10, max_err * 2 + 5)
    prev = [0] * (n + 1)
    curr = [INF] * (n + 1)
    
    for j in range(n + 1):
        prev[j] = 0
    
    for i in range(1, m + 1):
        char_code = read_bytes[i-1]
        j0 = max(1, i - band)
        j1 = min(n, i + band)
        
        if j0 > j1:
            return INF, region_start, region_start
        
        curr[0] = i   
        prev_row = prev
        for j in range(j0, j1 + 1):
            cost = 0 if char_code == ref_bytes[j-1] else 1
            curr[j] = min(prev_row[j-1] + cost, 
                          prev_row[j] + 1, 
                          curr[j-1] + 1)
        
        slice_vals = curr[j0:j1+1]
        if not slice_vals:
            return INF, region_start, region_start
        elif i > 10 and min(slice_vals) > max_err:
            return INF, region_start, region_start
            
        prev = list(curr)
        curr = [INF] * (n + 1)

    check_start = max(1, m - band)
    check_end = min(n, m + band)
    
    if check_start > check_end:
        return INF, region_start, region_start
    
    valid_scores = prev[check_start : check_end + 1]
    if not valid_scores:
        return INF, region_start, region_start
    
    best_cost = min(valid_scores)
    best_j = check_start + valid_scores.index(best_cost)
    start_pos = region_start + max(0, best_j - m)
    end_pos = region_start + best_j
    return best_cost, start_pos, end_pos


def map_reads(fm,
              reads,
              params
              ) -> List[Tuple[str, int, int]]:
    results = []
    ref_len = fm['n'] - 1
    
    for i, (rid, seq) in enumerate(reads):
        if i % 100 == 0:
            print(f"Mapping {i}/{len(reads)}...", end='\r')
        
        s = seq
        best_hit = None
        best_cost = INF
        max_err = max(5, int(len(s) * params['error_rate']))
        candidates = gather_candidates(fm, s, k=params['k'], step=params['step'], 
                                     max_pos_per_seed=params['max_seed_hits'], top_n=params['top_n'])

        if not candidates:
            retry_step = max(5, params['k'] // 2)
            candidates = gather_candidates(fm, s, k=params['k'], step=retry_step, 
                                         max_pos_per_seed=params['max_seed_hits'], top_n=params['top_n'])
            
        if not candidates: continue

        for off, _, _ in candidates:
            region_start = max(0, off - params['slack'])
            region_end = min(ref_len, off + len(s) + params['slack'])
            if region_start >= region_end: continue
            
            ref_sub = fm['text_bytes'][region_start:region_end]
            cost, st, en = banded_align(s, ref_sub, region_start, max_err)
            
            if cost < best_cost:
                best_cost = cost
                best_hit = (rid, st, en, cost)
       
        if best_hit is not None and best_cost <= int(len(s) * params['error_rate']):
            rid_out = best_hit[0]
            start1 = best_hit[1] + 1
            end1 = best_hit[2] + 1
            results.append((rid_out, start1, end1))
            
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("reads")
    parser.add_argument("output")
    args = parser.parse_args()

    monitor = MemoryMonitor()
    monitor.start()

    try:
        print("Reading Reference...")
        ref_recs = list(SeqIO.parse(args.reference, 'fasta'))
        if not ref_recs: return
        ref = "".join(str(r.seq).upper() for r in ref_recs)
        
        print(f"Building FM-Index (len={len(ref)})...")
        t0 = time.time()
        fm = build_fm_index(ref)
        print(f"Built in {time.time()-t0:.2f}s")
        
        reads = [(r.id, str(r.seq).upper()) for r in SeqIO.parse(args.reads, 'fasta')]
        print(f"Loaded {len(reads)} reads.")
        
        params = {
            'k': PARAM_K,
            'step': PARAM_STEP,
            'slack': PARAM_SLACK,
            'max_seed_hits': PARAM_MAX_HITS,
            'error_rate': PARAM_MAX_ERR,
            'top_n': PARAM_TOP_N
        }
        
        t1 = time.time()
        results = map_reads(fm, reads, params)
        print(f"\nDone in {time.time()-t1:.2f}s. Mapped: {len(results)}")
        
        with open(args.output, 'w') as f:
            for r in results: f.write(f"{r[0]}\t{r[1]}\t{r[2]}\n")
                
    finally:
        gc.collect()
        monitor.stop()
        monitor.join()
        monitor.plot()

if __name__ == '__main__':
    main()
