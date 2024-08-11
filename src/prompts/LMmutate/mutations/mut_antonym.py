import json
from typing import List
from ..mutations import PROMPT
from ..config.chatglm_config import LLM
from .prompt_helper import get_all_mut_info


def antonym(sentence: str, instance: List[str]):
    mutation_type = "Antonym mutation"
    type_description = "Antonym mutation involves replacing a word in the original sentence with its opposite or a word of the same type that has a contradictory meaning."
    prompt_get_mut_type_and_sentence = PROMPT.format(mutation_type, type_description,instance, sentence)
    output_string = LLM(prompt_get_mut_type_and_sentence).replace("\n", "")

    return get_all_mut_info(sentence,instance,output_string)
