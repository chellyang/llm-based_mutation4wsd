import json
from typing import List
from ..mutations import PROMPT
from ..config.chatglm_config import LLM
from .prompt_helper import get_all_mut_info


def tenseplus(sentence: str, instance: List[str]):
    mutation_type = "Tense mutation"
    type_description = "(1) Tense mutation refers to the change of the occurrence timeframe of an action or state within a specified range. " \
                       "(2) There are only 8 types of tenses involved: Simple Present Tense, Simple Past Tense, Simple Future Tense, Future in the Past Tense, Present Continuous Tense, Past Continuous Tense, Present Perfect Tense, and Past Perfect Tense."
    prompt_get_mut_type_and_sentence = PROMPT.format(mutation_type, type_description,instance, sentence)
    output_string = LLM(prompt_get_mut_type_and_sentence).replace("\n", "")

    return get_all_mut_info(sentence, instance, output_string)
