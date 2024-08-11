import json
from typing import List
from ..mutations import PROMPT
from ..config.chatglm_config import LLM
from .prompt_helper import get_all_mut_info


def number(sentence: str, instance: List[str]):
    mutation_type = "Numerical Quantifier mutation"
    type_description = "Numerical Quantifier Variation involves replacing the original numerals and quantifiers in a sentence with other suitable numerals or similar quantifiers."
    prompt_get_mut_type_and_sentence = PROMPT.format(mutation_type, type_description,instance, sentence)
    output_string = LLM(prompt_get_mut_type_and_sentence).replace("\n", "")

    return get_all_mut_info(sentence,instance,output_string)
