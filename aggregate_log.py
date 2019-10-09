import argparse

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', required=True, help='Input log file')
    parser.add_argument('-o', required=True, help='Output file')
    args = parser.parse_args()

    assert args.f != args.o

    data = open(args.f).readlines()
    with open(args.o, 'w') as f:
        f.write(args.f + '\n')
        f.write("epoch\titeration\tepoch x iter\tloss\tppl\tlr\n")
        for line in data:
            if line[:7] != '| epoch':
                continue
            line = line.replace('\n', '')

            epoch = int(line[8:11])

            iteration = line[14:21].replace('/', '').replace(' ', '')
            try:
                iteration = int(iteration)
            except ValueError:
                print('Error iteration', iteration)
                continue

            max_iter = line[22:30].replace('l', '')
            try:
                max_iter = int(max_iter)
            except ValueError:
                print('Error max_iter', max_iter)
                continue

            epoch_iter = (epoch - 1) * max_iter + iteration

            loss = line[line.find('loss=')+5:line.find('loss=')+10]
            try:
                loss = float(loss)
            except ValueError:
                print('Error loss', loss)
                continue

            ppl = line[line.find('ppl=')+4:line.find('ppl=')+8]
            try:
                ppl = float(ppl)
            except ValueError:
                print('Error ppl', ppl)
                continue

            lr = line[line.find('lr=')+3:line.find('lr=')+14].split(',')[0]
            try:
                lr = float(lr)
            except ValueError:
                print('Error lr', lr)
                continue

            f.write(f"{epoch}\t{iteration}\t{epoch_iter}\t{loss}\t{ppl}\t{lr}\n")


if __name__ == '__main__':
    main()

