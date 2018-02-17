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

def clean_text(content):
    content = content.replace("\n", " ")
    content = textacy.preprocess_text(content, lowercase=True, no_contractions=True)
    return content

