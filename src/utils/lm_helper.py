import spacy
from spacy.tokenizer import Tokenizer
from spacy.symbols import ORTH
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
from spacy.tokens import Doc
import stanza
from stanza.server import CoreNLPClient


SEP = '@#*'

class SpecialTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(SEP)
        return Doc(self.vocab, words=words)


def custom_tokenizer(nlp):
    # unify tokenization method as much as possible
    special_cases = {"``": [{ORTH: "``"}]}
    infixes = nlp.Defaults.prefixes + (r"[./]", r"[-]~", r"(.'.)")
    infix_re = spacy.util.compile_infix_regex(infixes)
    return Tokenizer(nlp.vocab, infix_finditer=infix_re.finditer, rules=special_cases)


def load_model():
    en_nlp = spacy.load('en_core_web_sm')
    en_nlp.tokenizer = SpecialTokenizer(en_nlp.vocab)
    # en_nlp.tokenizer.add_special_case("``", [{ORTH: "``"}])
    return en_nlp

    
# VERB - verbs (all tenses and modes)
# NOUN - nouns (common and proper)
# PRON - pronouns
# ADJ - adjectives
# ADV - adverbs
# ADP - adpositions (prepositions and postpositions)
# CONJ - conjunctions
# DET - determiners
# NUM - cardinal numbers
# PRT - particles or other function words
# X - other: foreign words, typos, abbreviations
# . - punctuation

def wordnet_pos_code(tag):
    if tag in ["NOUN"]:
        return wn.NOUN
    elif tag in ["VERB"]:
        return wn.VERB
    elif tag in ["ADJ"]:
        return wn.ADJ
    elif tag in ["ADV"]:
        return wn.ADV
    else:
        return None


def nltk_parser(tokens):        
    lemmatizer = WordNetLemmatizer()
    # choose universal tagset to match xml file
    tagged = nltk.pos_tag(tokens, tagset='universal')
    tokens_pos_lemma = []
    for word, tag in tagged:
        wntag = wordnet_pos_code(tag)
        if wntag is None:# not supply tag in case of None
            lemma = lemmatizer.lemmatize(word.lower()) 
        else:
            lemma = lemmatizer.lemmatize(word.lower(), pos=wntag)
        tokens_pos_lemma.append((word, tag, lemma)) 
    return tokens_pos_lemma

# FIXME, pos_tag with default tokenization
def corenlp_parser(sep_tokenized_text):
    whitespace_tokenized_text = sep_tokenized_text.replace(SEP, " ")
    with CoreNLPClient(
            annotators=['tokenize', 'ssplit', 'pos','lemma'],
            timeout=10000,
            memory='6G') as client:
        ann = client.annotate(whitespace_tokenized_text)
        tokens_pos_lemma = []
        for token in ann.sentence:
            print(token)
        return tokens_pos_lemma


if __name__ == '__main__':
    text = "change-ringing"
    # text = "A"+SEP+"large number"+SEP+"of"+SEP+"studies"+SEP+"with"+SEP+"Cerenia"+SEP+"were"+SEP+"carried"+SEP+"out"
    print(corenlp_parser(text))