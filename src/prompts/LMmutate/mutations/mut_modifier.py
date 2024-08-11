import json
from typing import List
from ..mutations import PROMPT
from ..config.chatglm_config import LLM
from .prompt_helper import get_all_mut_info


def modifier(sentence: str, instance: List[str]):
    mutation_type = "Modifier mutation"
    type_description = "Modifier mutation is adding an adverb or adjective in the appropriate position in the original sentence without changing the existing modifiers.  "
    prompt_get_mut_type_and_sentence = PROMPT.format(mutation_type, type_description,instance, sentence)
    output_string = LLM(prompt_get_mut_type_and_sentence).replace("\n", "")

    return get_all_mut_info(sentence, instance, output_string)
