import json
from typing import List
from ..mutations import PROMPT
from ..config.chatglm_config import LLM
from .prompt_helper import get_all_mut_info


def passivity(sentence: str, instance: List[str]):
    mutation_type = "Voice mutation"
    type_description = "Voice alternation changes the original sentence from the active voice to the passive voice, or from the passive voice to the active voice. "
    prompt_get_mut_type_and_sentence = PROMPT.format(mutation_type, type_description,instance, sentence)
    output_string = LLM(prompt_get_mut_type_and_sentence).replace("\n", "")

    return get_all_mut_info(sentence, instance, output_string)
