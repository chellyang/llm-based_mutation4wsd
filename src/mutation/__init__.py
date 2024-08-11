

from enum import Enum
from http.client import CONFLICT

# 代码功能：MutationResult是一个枚举类，用于定义特定的变异结果类型。
# 枚举类中包含了四个枚举值，分别表示"变异"、“丢弃”、“冲突"和"NER（命名实体识别）”。
class MutationResult(Enum):
    MUTATED = "MUTATED"
    DUMP = "DUMP"
    CONFLICT = "CONFLICT"
    NER = "NER"