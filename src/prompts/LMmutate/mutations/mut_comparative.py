import json
from typing import List
from ..mutations import PROMPT
from ..config.chatglm_config import LLM
from .prompt_helper import get_all_mut_info


def comparative(sentence: str, instance: List[str]):
    mutation_type = "Comparative mutation"
    type_description = "(1) Comparative mutation involves replacing the comparative structure of the original sentence with the superlative, or replacing the superlative structure with the comparative;" \
                       "(2) Comparative mutation does not apply to simple sentence structures."
    prompt_get_mut_type_and_sentence = PROMPT.format(mutation_type, type_description,instance, sentence)
    output_string = LLM(prompt_get_mut_type_and_sentence).replace("\n", "")

    return get_all_mut_info(sentence, instance, output_string)
