from clean import *
from textacy import preprocess as pp

def preprocess(text, fix_unicode=True, normalize_white_space = False, lowercase=False, transliterate=False,
                    no_urls=False, no_emails=False, no_phone_numbers=False,
                    no_numbers=False, no_currency_symbols=False, no_punct=False,
                    no_contractions=False, no_accents=False):
    if normalize_white_space:
        text = pp.normalize_whitespace(text)
    text = pp.preprocess_text(text, fix_unicode, lowercase, transliterate,
                    no_urls, no_emails, no_phone_numbers,
                    no_numbers, no_currency_symbols, no_punct,
                    no_contractions, no_accents)
    return text

def split_sentence(paragraph):
    paragraph = re.split(pattern="([a-zA-Z\(\)]{2,}[.?!])\s+", string=paragraph)
    paragraph = [a + b for a, b in itertools.zip_longest(paragraph[::2], paragraph[1::2], fillvalue='')]
    if paragraph:
        paragraph = [clean_str(e) for e in paragraph]
    return paragraph