import logging

import textacy
import itertools
import re

# bad unicode fix function clean('../data/MLP-400AV/YES/yee_whye_teh/yee_whye_teh_2_1.txt')
def clean(fname):
    text = textacy.fileio.read.read_file(fname)
    text = textacy.preprocess_text(fix_unicode=True, no_accents=True, transliterate=True, text=text)
    text = textacy.preprocess.fix_bad_unicode(text=text)
    text = text.replace('?', 'f')
    with open(fname + '.txt', 'w') as f:
        f.write(text)

def clean_str(str):
    # string = re.sub("\'", " \' ", string)
    # string = re.sub("\"", " \" ", string)
    # string = re.sub("-", " - ", string)

    # string = re.sub(",", " , ", string)

    # string = re.sub(r"[(\[{]", " ( ", string)
    # string = re.sub(r"[)\]}]", " ) ", string)
    # string = re.sub("\s{2,}", " ", string)

    return str.strip()


def clean_text_minor(content:str):
    logging.info("Minor Text Cleanup: LOWERCASE, NO_CONTRACTION")

    content = content.replace("\n", " ")
    content = textacy.preprocess_text(content,
                                      lowercase=True,
                                      no_contractions=True)

    # content = re.sub("\'", " \' ", content)
    # content = re.sub("\"", " \" ", content)

    return content


def clean_text_major(content):
    logging.info("Minor Text Cleanup: UNICODE, LOWERCASE, TRANSLITERATE, NO_NUM, NO_CONTRACTION")
    content = content.replace("\n", " ")
    content = textacy.preprocess_text(content,
                                      fix_unicode=True,
                                      lowercase=True,
                                      transliterate=True,
                                      no_numbers=True,
                                      no_contractions=True)

    return content
