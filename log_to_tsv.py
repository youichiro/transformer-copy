import re
import argparse

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', required=True, help='Input log file')
    parser.add_argument('-o', required=True, help='Output file')
    args = parser.parse_args()

    assert args.f != args.o
    pattern = r"^\| epoch (.*):(.*)/(.*) loss=(.*), dustribution_loss=(.*), label_loss=(.*), copy_alpha=(.*), ppl=(.*), wps=.*$"

    data = open(args.f).readlines()
    with open(args.o, 'w') as f:
        f.write(args.f + '\n')
        f.write("epoch\titeration\tepoch x iter\tloss\tdistribution_loss\tlabel_loss\tcopy_alpha\tppl\n")
        for line in data:
            line = line.replace('\n', '')
            if not line.startswith('| epoch') or 'valid on' in line:
                continue
            m = re.match(pattern, line)
            if not m:
                continue
            epoch, ite, max_ite, loss, dist_loss, label_loss, copy_alpha, ppl = m.groups()
            epoch = int(epoch)
            ite = int(ite)
            max_ite = int(max_ite)
            epoch_ite = (epoch - 1) * max_ite + ite

            f.write(f"{epoch}\t{ite}\t{epoch_ite}\t{loss}\t{dist_loss}\t{label_loss}\t{copy_alpha}\t{ppl}\n")


if __name__ == '__main__':
    main()

