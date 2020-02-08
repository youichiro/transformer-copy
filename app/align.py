import collections
from itertools import groupby
from pprint import pprint


Trace = collections.namedtuple("Trace", ["cost", "ops"])


def TRANSPOSITION(A, B):
    cost = len(A) - 1
    return cost


class WagnerFischer:
    def __init__(self, src: list, tgt: list):
        self.len_s = len(src)
        self.len_t = len(tgt)
        self._table = [[None for _ in range(self.len_t + 1)] for
                       _ in range(self.len_s + 1)]
        self._table[0][0] = Trace(0, {"O"})  # start cell
        for i in range(1, self.len_s + 1):
            self._table[i][0] = Trace(self._table[i-1][0].cost + 1, {"D"})
        for j in range(1, self.len_t + 1):
            self._table[0][j] = Trace(self._table[0][j-1].cost + 1, {"I"})

        for i in range(self.len_s):
            for j in range(self.len_t):
                if src[i] == tgt[j]:
                    self._table[i+1][j +
                                     1] = Trace(self._table[i][j].cost, {"M"})
                else:
                    cost_D = self._table[i][j+1].cost + 1
                    cost_I = self._table[i+1][j].cost + 1
                    cost_S = self._table[i][j].cost + 1
                    cost_T = float("inf")
                    min_val = min(cost_D, cost_I, cost_S)

                    k = 1
                    while i > 0 and j > 0 and (i - k) >= 0 and (j - k) >= 0 \
                            and self._table[i-k+1][j-k+1].cost - self._table[i-k][j-k].cost > 0:
                        if collections.Counter(src[i-k:i+1]) == collections.Counter(tgt[j-k:j+1]):
                            cost_T = self._table[i-k][j-k].cost + \
                                TRANSPOSITION(src[i-k:i+1], tgt[j-k:j+1])
                            min_val = min(min_val, cost_T)
                            break
                        k += 1

                    trace = Trace(min_val, [])
                    if cost_D == min_val:
                        trace.ops.append("D")
                    if cost_I == min_val:
                        trace.ops.append("I")
                    if cost_S == min_val:
                        trace.ops.append("S")
                    if cost_T == min_val:
                        trace.ops.append("T" + str(k+1))
                    self._table[i+1][j+1] = trace

        self.cost = self._table[-1][-1].cost

    def _stepback(self, i, j, trace, path_back):
        for op in trace.ops:
            if op == "M":
                yield i-1, j-1, self._table[i-1][j-1], path_back + ["M"]
            elif op == "I":
                yield i, j-1, self._table[i][j-1], path_back + ["I"]
            elif op == "D":
                yield i-1, j, self._table[i-1][j], path_back + ["D"]
            elif op == "S":
                yield i-1, j-1, self._table[i-1][j-1], path_back + ["S"]
            elif op.startswith("T"):
                k = int(op[1:] or 2)
                yield i-k, j-k, self._table[i-k][j-k], path_back + [op]
            elif op == "O":
                return
            else:
                raise ValueError("Unknown op {!r}".format(op))

    def alignments(self, dfirst=False):
        """
        I: insertion, D: deletion, S: substitution, T: transposition, M: nothing
        """
        if dfirst:
            return self._dfirst_alignments()
        else:
            return self._bfirst_alignments()

    def _dfirst_alignments(self):
        stack = list(self._stepback(
            self.len_s, self.len_t, self._table[-1][-1], []))
        while stack:
            (i, j, trace, path_back) = stack.pop()
            if trace.ops == {"O"}:
                yield path_back[::-1]
                continue
            stack.extend(self._stepback(i, j, trace, path_back))

    def _bfirst_alignments(self):
        queue = collections.deque(self._stepback(
            self.len_s, self.len_t, self._table[-1][-1], []))
        while queue:
            (i, j, trace, path_back) = queue.popleft()
            if trace.ops == {"O"}:
                yield path_back[::-1]
                continue
            queue.extend(self._stepback(i, j, trace, path_back))


def get_opcodes(alignment):
	s_start = s_end = t_start = t_end = 0
	opcodes = []
	for op in alignment:
		if op[0] == "D":  # Deletion
			s_end += 1
		elif op[0] == "I":  # Insertion
			t_end += 1
		elif op[0].startswith("T"):  # Transposition
			k = int(op[1:] or 2)
			s_end += k
			t_end += k
		else:  # Match or substitution
			s_end += 1
			t_end += 1
		opcodes.append((op, s_start, s_end, t_start, t_end))
		s_start = s_end
		t_start = t_end
	return opcodes


def merge_edits(edits):
	if edits:
		return [("X", edits[0][1], edits[-1][2], edits[0][3], edits[-1][4])]
	else:
		return edits


def get_edits_group_type(edits):
	new_edits = []
	for op, group in groupby(edits, lambda x: x[0]):
		if op != "M":
			 new_edits.extend(merge_edits(list(group)))
	return new_edits


def get_edits_group_all(edits):
	new_edits = []
	for op, group in groupby(edits, lambda x: True if x[0] == "M" else False):
		if not op:
			 new_edits.extend(merge_edits(list(group)))
	return new_edits


def get_aligned_edits(src_raw: str, tgt_raw: str):
    src = ' '.join(src_raw.replace(' ', '')).split(' ')
    tgt = ' '.join(tgt_raw.replace(' ', '')).split(' ')
    alignments = WagnerFischer(src, tgt)
    alignment = next(alignments.alignments(dfirst=True))
    edits = get_edits_group_all(get_opcodes(alignment))
    edit_dict = {
        'src_raw': src_raw,
        'tgt_raw': tgt_raw,
        'edits': [],
        'src_edit_pos': [],
        'tgt_edit_pos': [],
    }
    for i, edit in enumerate(edits):
        src_start, src_end, tgt_start, tgt_end = edit[1:5]
        src_str = ' '.join(src[src_start:src_end])
        tgt_str = ' '.join(tgt[tgt_start:tgt_end])
        if src_str and tgt_str:
            operation = 'substitution'
        elif src_str and not tgt_str:
            operation = 'deletion'
        elif not src_str and tgt_str:
            operation = 'insertion'
        else:
            operation = None

        edit_dict['edits'].append({
            "edit_id": i + 1,
            "src_start": src_start,
            "src_end": src_end,
            "tgt_start": tgt_start,
            "tgt_end": tgt_end,
            "src_str": src_str,
            "tgt_str": tgt_str,
            "operation": operation,
        })
        edit_dict['src_edit_pos'].extend(list(range(src_start, src_end)))
        edit_dict['tgt_edit_pos'].extend(list(range(tgt_start, tgt_end)))
    return edit_dict


if __name__ == '__main__':
    src = "今日は車を買う。"
    tgt = "今日は車まで買う"
    edits = get_aligned_edits(src, tgt)
    pprint(edits)
