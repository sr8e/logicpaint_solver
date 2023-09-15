import math
import pathlib
from collections import deque
from typing import NamedTuple


class LogicPaintError(Exception):
    pass


class LogicPaintContradictError(LogicPaintError):
    pass


class LogicPaintLoadError(LogicPaintError):
    pass


class LogicPaintTable:
    class ColumnSplit(NamedTuple):
        # a split part of column or row
        part: list
        start_index: int

        def __len__(self) -> int:
            return len(self.part)

        def replace(self, part):
            assert len(part) == len(self.part)
            return self.__class__(part, self.start_index)

        @property
        def complete(self) -> bool:
            return all(v != -1 for v in self.part)

    def __init__(self, w, h, row_cond, col_cond):
        self.table = [[-1] * w] * h
        self.geometry = (w, h)
        self.cond = (row_cond, col_cond)
        assert (
            len(row_cond) == h and len(col_cond) == w
        ), f"{len(row_cond)} != {h} or {len(col_cond)} != {w}"

    def get_col(self, index, axis):
        if axis == 0:
            return self.table[index][:]
        else:
            return [row[index] for row in self.table]

    def set_col(self, index, axis, arr):
        assert len(arr) == self.geometry[axis]
        if axis == 0:
            self.table[index] = arr
        else:
            for i, v in enumerate(arr):
                self.table[i][index] = v

    def print_table(self, fill="x", blank=" ", undef="_"):
        print_map = {-1: undef, 0: blank, 1: fill}
        for row in self.table:
            print("".join([print_map[v] for v in row]))

    @classmethod
    def load(cls, path_str):
        def read_parse(f):
            return [int(s) for s in f.readline().split()]

        path = pathlib.Path(path_str)
        if not (path.exists() and path.is_file()):
            raise LogicPaintLoadError("Specified path is not a file or does not exist.")

        with open(path.absolute(), "r") as f:
            try:
                w, h = read_parse(f)
                row_cond = [read_parse(f) for _ in range(h)]
                col_cond = [read_parse(f) for _ in range(w)]
            except:
                raise LogicPaintLoadError("Loaded file is not in correct format.")
        return cls(w, h, row_cond, col_cond)

    @classmethod
    def split_column(cls, arr):
        split = []
        start_index = 0
        for i in range(len(arr)):
            if arr[i] != 0:
                continue
            if i == start_index:
                # consecutive 0
                start_index += 1
                continue
            split.append(cls.ColumnSplit(arr[start_index:i], start_index))
            start_index = i + 1
        if start_index != len(arr):
            split.append(cls.ColumnSplit(arr[start_index : len(arr)], start_index))
        return split

    @staticmethod
    def enumerate_split(cond, num):
        split = []
        cond_len = len(cond)

        class SplitEnumState(NamedTuple):
            set_index: int
            cond_index: int
            cur_split: list

        queue = deque()
        queue.append(SplitEnumState(0, 0, [()]))

        while len(queue) > 0:
            state = queue.popleft()
            s = state.set_index
            c = state.cond_index
            if c == cond_len:
                res = state.cur_split + [()] * (num - s - 1)
                assert len(res) == num
                split.append(res)
                continue
            for i in range(num - s):
                next_split = state.cur_split[:]
                if i == 0:
                    next_split[s] = state.cur_split[s] + (cond[c],)
                else:
                    next_split += [()] * (i - 1) + [(cond[c],)]
                queue.append(SplitEnumState(s + i, c + 1, next_split))

        assert len(split) == math.comb(len(cond) + num - 1, num - 1)
        return split

    @staticmethod
    def get_max_conseq(arr):
        seq = 0
        maxseq = 0
        for v in arr:
            if v == 1:
                seq += 1
            else:
                maxseq = max(seq, maxseq)
                seq = 0
        return maxseq

    @classmethod
    def get_valid_split(cls, arr, cond):
        col_splits = cls.split_column(arr)
        col_split_lens = [len(s.part) for s in col_splits]
        col_split_seqs = [cls.get_max_conseq(s.part) for s in col_splits]
        cond_split_set = cls.enumerate_split(cond, len(col_splits))
        valid = []
        for cond_split in cond_split_set:
            for col_l, col_s, cond in zip(col_split_lens, col_split_seqs, cond_split):
                if col_l < sum(cond) + len(cond) - 1 or col_s > max(cond, default=0):
                    break
            else:
                valid.append(cond_split)
        return col_splits, valid

    @staticmethod
    def solve_split(col, cond):
        if col.part.count(1) == sum(cond):
            return col.replace([0 if v == -1 else v for v in col.part])
        ambi_val = len(col) - (sum(cond) + len(cond) - 1)

        index = 0
        res = col.part[:]
        for c in cond:
            if ambi_val < c:
                res[index + ambi_val : index + c] = [1] * (c - ambi_val)
                if ambi_val == 0 and index + c < len(col):
                    res[index + c] = 0
            index += c + 1
        # print(res)
        return col.replace(res)

    def solve_column(self, index, axis):
        arr = self.get_col(index, axis)
        col_splits, valid_split = self.get_valid_split(arr, self.cond[axis][index])

        if len(valid_split) == 0:
            raise LogicPaintContradictError()
        if len(valid_split) == 1:
            cond_split = valid_split[0]
            for col, cond in zip(col_splits, cond_split):
                # print(f"{axis=}, {index=}, {cond_split}, {col.part}, {col.complete}")
                if col.complete:
                    continue
                col_res = self.solve_split(col, cond)
                arr[col.start_index : col.start_index + len(col)] = col_res.part
        else:
            pass

        self.set_col(index, axis, arr)

    def solve(self):
        while True:
            self.print_table()
            for i, l in enumerate(self.geometry):
                for index in range(l):
                    self.solve_column(index, i)


logicpaint = LogicPaintTable.load("./testcase/1-001.txt")
logicpaint.solve()
