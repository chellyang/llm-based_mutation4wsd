import json
from typing import List
from ..mutations import PROMPT
from ..config.chatglm_config import LLM
from .prompt_helper import get_all_mut_info


def inversion(sentence: str, instance: List[str]):
    mutation_type = "Inversion mutation"
    type_description = "Inversion mutation can fully or partially invert sentences that can be inverted, and mutations " \
                       "do not occur in sentences that cannot be inverted or where inversion is unnatural."
    prompt_get_mut_type_and_sentence = PROMPT.format(mutation_type, type_description,instance, sentence)
    output_string = LLM(prompt_get_mut_type_and_sentence).replace("\n", "")

    return get_all_mut_info(sentence, instance, output_string)
