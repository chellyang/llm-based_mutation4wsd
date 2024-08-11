import json
from typing import List
from ..mutations import PROMPT
from ..config.chatglm_config import LLM
from .prompt_helper import get_all_mut_info


def demonstrative(sentence: str, instance: List[str]):
    mutation_type = "Demonstrative mutation"
    type_description = "Demonstrative pronoun mutation refers to replacing specific pronouns and proper nouns in the original sentence. " \
                       "This includes:(1) Replacing a specific pronoun with a proper noun in the context.(2) Replacing a proper noun with a suitable pronoun." \
                       "The replaceable pronouns include: this/that/these/those/it/they/them/us/you/me."
    prompt_get_mut_type_and_sentence = PROMPT.format(mutation_type, type_description,instance, sentence)
    output_string = LLM(prompt_get_mut_type_and_sentence).replace("\n", "")

    return get_all_mut_info(sentence, instance, output_string)
