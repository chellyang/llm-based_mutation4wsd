

import copy
from typing import List, Optional, Tuple

# 代码功能：MutationSentence类是一个用于处理语句变异的工具类。
# 它包含了插入和删除标记，并提供了获取结果、获取某个索引的元素、设置某个索引的值以及获取长度的功能。
# 该类的实例初始化时需要传入一个tokens列表，然后可以通过调用delete_token方法来标记要删除的标记位置，
# 通过调用insert_tokens方法来插入新的标记位置，最后可以调用get_result方法获取修改后的tokens列表以及对齐信息的列表。
# 此外，还重载了获取索引运算符（getitem）、设置索引运算符（setitem）和len()函数，以方便对实例进行操作。
# 定义一个名为MutationSentence的类
class MutationSentence(object):
    def __init__(self, tokens: List[str]) -> None:
        self._tokens = copy.deepcopy(tokens)
        self._insert_tokens: List[Optional[List[str]]] = [None] * (len(tokens) + 1)
        self._delete_tokens: List[Optional[bool]] = [False] * len(tokens)

    def delete_token(self, index: int):
        self._delete_tokens[index] = True

    def insert_tokens(self, index: int, tokens: List[str]):
        if self._insert_tokens[index] is None:
            self._insert_tokens[index] = []
        self._insert_tokens[index].extend(tokens)

    def get_result(self) -> Tuple[List[str], List[Tuple[int, int]]]:
        ret = []
        align = []

        for i in range(len(self._tokens)):
            if self._insert_tokens[i] is not None:
                ret.extend(self._insert_tokens[i])
            if not self._delete_tokens[i]:
                align.append((i, len(ret)))
                ret.append(self._tokens[i])

        return ret, align

    def __getitem__(self, key):
        return self._tokens[key]

    def __setitem__(self, key, newvalue):
        self._tokens[key] = newvalue

    def __len__(self) -> int:
        return len(self._tokens)
            