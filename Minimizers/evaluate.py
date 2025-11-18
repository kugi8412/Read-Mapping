# evaluate.py
# -*- coding: utf-8 -*-


import utils
import argparse


if __name__ == '__main__':
    print("Evaluating mapping results with minimizers...")

    parser = argparse.ArgumentParser()
    parser.add_argument('mapper_output', type=str)
    parser.add_argument('reference', type=str)
    args = parser.parse_args()

    mapped_reads, acc = utils.mapped_reads_and_acc(args.mapper_output, args.reference)
    print(f'{mapped_reads:.4f} of all reads mapped with accuracy {acc:.4f}')
