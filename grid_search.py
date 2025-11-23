# grid_search.py
# -*- coding: utf-8 -*-

import os
import csv
import time
import argparse
import itertools

from typing import Dict, List, Tuple

import utils
import mapper

INF = 10**9
PARAM_K = 16
PARAM_STEP = 30
PARAM_MAX_ERR = 0.14
PARAM_SLACK = 200
PARAM_MAX_HITS = 1000
PARAM_TOP_N = 6


def run_single_setting(fm: Dict,
                       reads: List[Tuple[str, str]],
                       params: Dict,
                       out_prefix: str,
                       out_dir: str,
                       iter_id: int
                       ) -> Tuple[str, float]:
    """ Run mapping for one parameter setting.
    Returns tuple (output_file_path, elapsed_seconds)
    """
    start = time.perf_counter()
    mapped = mapper.map_reads(fm, reads, params)
    elapsed = time.perf_counter() - start

    # write mapping results to file (tab-separated: id, start, end)
    out_filename = f"{out_prefix}_iter{iter_id}_k{params['k']}_step{params['step']}_slack{params['slack']}_hits{params['max_seed_hits']}_err{params['error_rate']}_top{params['top_n']}.txt"
    out_path = os.path.join(out_dir, out_filename)
    with open(out_path, 'w') as of:
        for rid, s, e in mapped:
            of.write(f"{rid}\t{s}\t{e}\n")
    return out_path, elapsed

def append_to_csv(csv_path: str, row: Dict):
    header = ['timestamp','iter_id','k','step','slack','max_seed_hits','error_rate','top_n',
              'time_sec','mapped_fraction','accuracy','out_file']
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def grid_search(reference: str,
                reads_fasta: str,
                truth_file: str,
                out_dir: str,
                results_csv: str,
                grid_params: Dict[str, List],
                base_params: Dict,
                serialize_index: str = None):
    os.makedirs(out_dir, exist_ok=True)

    print("Reading reads...")
    reads = mapper.read_reads(reads_fasta)
    print(f"Reads count: {len(reads)}")

    print("Building FM-index (once)...")
    t0 = time.perf_counter()
    fm = mapper.build_fm_index(reference)
    t_build = time.perf_counter() - t0
    print(f"Built FM-index in {t_build:.2f}s (n={fm['n']})")

    combos = list(itertools.product(*[grid_params[k] for k in sorted(grid_params.keys())]))
    param_keys = sorted(grid_params.keys())
    print(f"Total combinations: {len(combos)}")

    iter_id = 0
    for combo in combos:
        iter_id += 1
        params = base_params.copy()
        params.update({k: v for k, v in zip(param_keys, combo)})
        print(f"[{iter_id}/{len(combos)}] Running params: {params}")
        try:
            out_file, elapsed = run_single_setting(fm, reads, params, out_prefix="mapping", out_dir=out_dir, iter_id=iter_id)
        except Exception as e:
            print(f"ERROR on iteration {iter_id} with params {params}: {e}")
            row = {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'iter_id': iter_id,
                'k': params.get('k'),
                'step': params.get('step'),
                'slack': params.get('slack'),
                'max_seed_hits': params.get('max_seed_hits'),
                'error_rate': params.get('error_rate'),
                'top_n': params.get('top_n'),
                'time_sec': None,
                'mapped_fraction': None,
                'accuracy': None,
                'out_file': f"ERROR: {e}"
            }
            append_to_csv(results_csv, row)
            continue

        try:
            mapped_frac, acc = utils.mapped_reads_and_acc(out_file, truth_file)
        except Exception as e:
            print(f"Could not compute accuracy for {out_file}: {e}")
            mapped_frac, acc = None, None

        row = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'iter_id': iter_id,
            'k': params.get('k'),
            'step': params.get('step'),
            'slack': params.get('slack'),
            'max_seed_hits': params.get('max_seed_hits'),
            'error_rate': params.get('error_rate'),
            'top_n': params.get('top_n'),
            'time_sec': round(elapsed, 3),
            'mapped_fraction': mapped_frac,
            'accuracy': acc,
            'out_file': out_file
        }
        append_to_csv(results_csv, row)
        print(f" -> time: {elapsed:.2f}s, mapped_frac: {mapped_frac}, acc: {acc}, out: {out_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run grid search for mapper and log time + accuracy to CSV")
    parser.add_argument("--reference", required=True, help="Reference FASTA (string, not path to module). If you want to pass a file, load it first and supply its sequence.")
    parser.add_argument("--reads", required=True, help="Reads FASTA")
    parser.add_argument("--truth", required=True, help="Truth mapping file (tab: id start end) to compute accuracy")
    parser.add_argument("--out-dir", default="grid_out", help="Directory to store mapping outputs")
    parser.add_argument("--results-csv", default="grid_results.csv", help="CSV file to append results")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    reference_seq = mapper.read_fasta_concat(args.reference)

    # example grid parameters
    grid_params = {
        'k': [15, 16, 18],                     # k-mer sizes
        'step': [26, 30, 34],                  # seed step
        'slack': [100, 200, 300],              # slack around candidate
        'max_seed_hits': [800, 1000, 1200],    # limit seed occurrences
        'error_rate': [0.12, 0.15],            # allowed error rate
        'top_n': [5, 6]                        # number of top offsets to try
    }

    base_params = {
        'k': PARAM_K,
        'step': PARAM_STEP,
        'slack': PARAM_SLACK,
        'max_seed_hits': PARAM_MAX_HITS,
        'error_rate': PARAM_MAX_ERR,
        'top_n': PARAM_TOP_N
    }

    grid_search(reference_seq, args.reads, args.truth, args.out_dir, args.results_csv, grid_params, base_params)
