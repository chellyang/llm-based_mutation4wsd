import json
from typing import List
from ..mutations import PROMPT
from ..config.chatglm_config import LLM
from .prompt_helper import get_all_mut_info


def that_this(sentence: str, instance: str):
    mutation_type = "Pronoun mutation"
    type_description = "Pronoun mutation replaces a pronoun in the original sentence with its opposite counterpart. " \
                       "It includes:this/that, these/those, it/they, them/us, you/me."
    prompt_get_mut_type_and_sentence = PROMPT.format(mutation_type, type_description,instance, sentence)
    output_string = LLM(prompt_get_mut_type_and_sentence).replace("\n", "")

    return get_all_mut_info(sentence, instance, output_string)
