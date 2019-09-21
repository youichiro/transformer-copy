# -*- coding: utf-8 -*-
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', required=True, help='Input file')
    parser.add_argument('-o', required=True, help='Output file')
    args = parser.parse_args()

    data = open(args.f).readlines()
    with open(args.o, 'w') as f:
        for line in tqdm(data):
            line = line.replace('\n', '').replace(' ', '')
            line = ' '.join(line)
            line = line.replace('< u n k >', '<unk>')
            f.write(line + '\n')


if __name__ == '__main__':
    main()
