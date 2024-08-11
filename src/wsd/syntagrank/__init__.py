import json
from typing import Dict, List
import requests


# Reference: https://github.com/pasinit/WSDFramework20
syntagrank_config = {
  "url": "http://api.syntagnet.org/",
  "disambiguate_text_endpoint": "disambiguate",
  "disambiguate_tokens_endpoint":  "disambiguate_tokens",
}

def disambiguate_tokens(tokens: List[Dict[str, str]], lang="EN", session=requests.Session()):
    """
    :param
        tokens: list of tokens. Each tokens has the information on the word, the lemma, the pos, the token_id
        and whether it is a target_word or not.
    :return
        a list of AnnotatedToken.
    """
    lang = lang.upper()
    url = syntagrank_config["url"] + syntagrank_config["disambiguate_tokens_endpoint"]
    id2token_idx = dict()
    for idx, token in enumerate(tokens):
        assert "lemma" in token
        assert "pos" in token
        assert "word" in token
        if "isTargetWord" not in token and "is_target_word" not in token:
            token["isTargetWord"] = False
        if "id" not in token:
            token["id"] = "None"
        if "is_target_word" in token:
            token["isTargetWord"] = token["is_target_word"]
            del token["is_target_word"]
        id2token_idx[token["id"]] = idx
    payload = {"lang": lang, "words": tokens}

    r = session.post(url, json=payload)

    response = json.loads(r.text)

    disambiguated_tokens = []
    for tagged_token in response["result"]:
        # synset here identifies the WordNet 3.0 offset for the concept assigned to the token. 
        disambiguated_tokens.append((tagged_token["id"], tagged_token["synset"]))
    return disambiguated_tokens


def test_disambiguate_tokens():
    lang = "it"
    tokens = [{"word": "questo", "lemma": "questo", "pos": "X"},
              {"word": "è", "lemma": "essere", "pos": "v", "id": "1", "isTargetWord":True},
              {"word": "un", "lemma":"un","pos":"X"},
              {"word": "semplice", "lemma":"semplice", "pos":"a", "id":"2", "isTargetWord":True},
              {"word": "testo", "lemma":"testo", "pos":"n", "id":"3", "isTargetWord":True},
              {"word": "d'", "lemma":"di", "pos":"X"},
              {"word": "esempio", "lemma":"esempio", "pos":"n", "id":"4", "isTargetWord":True}]
    tokens = disambiguate_tokens(tokens, lang=lang)
    print(tokens)             


if __name__ == "__main__":
    test_disambiguate_tokens()