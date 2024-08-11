

from typing import List, Union

# 代码功能：该函数用于在给定的original_deps和original_head中查找特定类型的子节点索引。
# 函数首先将type参数转换为列表形式，然后遍历original_deps列表，查找与指定类型匹配且与给定index关联的节点，并返回匹配的子节点索引。
# 如果未找到符合条件的索引，则返回-1。

# 定义一个函数，名为find_child_index，接收四个参数：original_deps为字符串列表，original_head为原始头部，index为整数，type为字符串或字符串列表，返回整数
def find_child_index(original_deps: List[str], original_head, index: int, type: Union[str, List]) -> int:
    # 初始化一个空列表types用于存储类型值
    types = []
    # 判断type是否为单个字符串，如果是，则将其添加到types列表中
    if isinstance(type, str):
        types.append(type)
    # 如果type不是单个字符串，而是字符串列表，则直接赋值给types
    else:
        types = type
    # 遍历original_deps列表的索引
    for i in range(len(original_deps)):
        # 判断条件，如果original_head[i]的值等于index，并且original_deps[i]在types列表中
        if original_head[i].i == index and original_deps[i] in types:
            # 如果条件成立，返回当前索引i
            return i
    # 如果未找到符合条件的索引，则返回-1
    return -1

def find_children_indices(original_deps: List[str], original_head, index: int, type: Union[str, List]) -> List[int]:
    types = []
    if isinstance(type, str):
        types.append(type)
    else:
        types = type
    
    ret = []
    for i in range(len(original_deps)):
        if original_head[i].i == index and original_deps[i] in types:
            ret.append(i)
    return ret

def recover_word(cleaned_word: str, original_word: str) -> str:
    '''
    cleaned_word: good
    original_word: Goods<Space>
    output: Good<Space>
    '''
    ret = cleaned_word
    if original_word[0].isupper():
        ret = ret.capitalize()
    
    if original_word.endswith(" ") and not ret.endswith(" "):
        ret = ret + " "
    
    return ret