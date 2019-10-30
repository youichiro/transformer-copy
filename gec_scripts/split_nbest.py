# argv[1]: nbestファイル名
# argv[2]: 出力ファイル名のpref
import sys

filename = sys.argv[1]
ofilepref = sys.argv[2]
src_ofilename = ofilepref + '.src'
tgt_ofilename = ofilepref + '.tgt'

lines = open(filename).readlines()
src_tokens = ''
tgt_tokens = ''
with open(src_ofilename, 'w') as fs, open(tgt_ofilename, 'w') as ft:
    for i, line in enumerate(lines):
        line = line.replace('\n', '')
        if line[0] == 'S':
            src_tokens = line.split('\t')[1]
        elif line[0] == 'H':
            tgt_tokens = line.split('\t')[2]
            fs.write(src_tokens + '\n')
            ft.write(tgt_tokens + '\n')
        else:
            pass

