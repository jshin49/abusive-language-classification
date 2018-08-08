import csv
import pandas as pd
import re

import nltk
from nltk.corpus import stopwords

import regex
import wordsegment as ws

FLAGS = re.MULTILINE | re.DOTALL


def re_sub(pattern, repl, text, flags=None):
    if flags is None:
        return re.sub(pattern, repl, text, flags=FLAGS)
    else:
        return re.sub(pattern, repl, text, flags=(FLAGS | flags))


def parse_sentences():
    '''
    since this is a multi-sentence document,
    it seems to make more sense to parse each data point into sentences,
    could use paragraph or skip-thought vectors
    '''
    pass


def clean_text(text,
               remove_nonalphanumeric,
               use_number_special_token,
               remove_numbers,
               deal_repetition,
               preserve_case,
               remove_hashtag_at_end,
               separate_contractions,
               separate_punctuations,
               remove_url,
               remove_stopwords,
               remove_nonlatin,
               row_id=None):

    if remove_url:
        text = re_sub(r"(http)\S+", "", text)
    else:
        text = re_sub(r"(http)\S+", "<url>", text)

    # ============== WIKIPEDIA related ==============

    # remove Image: keep name, remove extension
    # e.g. Image: 647147464205, 648330795032
    text = re_sub(r"(Image):(\S[^\.]+)(.\S+)", r"\2", text, re.IGNORECASE)

    # remove File Extensions :  .zip
    # e.g. 647162932442,
    # text = re_sub(r"(.*)\.[^.]{1,10}$", "", text)

    # remove user: keep name
    text = re_sub(r"User:", "", text, re.IGNORECASE)

    # remove WP: keep name
    text = re_sub(r"WP:", "", text, re.IGNORECASE)

    # remove Wikipedia: keep name
    text = re_sub(r"Wikipedia:", "", text, re.IGNORECASE)

    # Get rid of "User talk: ~"
    text = re_sub(r"User talk:", "", text, re.IGNORECASE)

    # remove time number:number replace with <time>
    text = re_sub(
        r"([0-9]+):([0-9]+)([a-z]){2}", "<time>", text, re.IGNORECASE)
    text = re_sub(r"([0-9]+):([0-9]+) (AM|PM)", "<time>", text, re.IGNORECASE)

    # remove date

    # Split camelcase words (2,3,4)
    # e.g. 242489483386
    # Using regex package instead of re package due to split error
    # text = " ".join(regex.split(
    #     r"([A-Z]+|[A-Z]?[a-z]+)(?=[A-Z]|\b)", text, flags=(regex.MULTILINE | regex.DOTALL | regex.VERSION1)))
    # Remove multiple spaces
    text = re_sub(r"[ \s\t\n]+", " ", text)

    # remove inline CSS tags
    # e.g.432442360332, 434747694335
    text = re_sub(r'\w+=\"\"[^\"]+\"\"', "", text)

    #### TOXIC COMMENT CORPUS SPECIFIC ####
    # NYScholar
    # Talkback, talkback

    #### TEST DATA SPECIFIC ####
    # remove WIKI_LINK: keep after @
    # remove EXTERNAL_LINK: keep name

    # ============== End of Wikipedia related ==============

    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if w not in stops]
        text = " ".join(text)

    if separate_contractions:
        #     text = re_sub(r"\'s", " \'s", text)
        #     text = re_sub(r"\'ve", " \'ve", text)
        text = re_sub(r"n\'t", " n\'t", text)
    #     text = re_sub(r"\'re", " \'re", text)
    #     text = re_sub(r"\'d", " \'d", text)
    #     text = re_sub(r"\'ll", " \'ll", text)

    if separate_punctuations:
        text = re_sub(r",", " , ", text)
        text = re_sub(r"!", " ! ", text)
        text = re_sub(r"\(", " ( ", text)
        text = re_sub(r"\)", " ) ", text)
        text = re_sub(r"\?", " ? ", text)

    # remove chinese character
    # e.g.435378044110
    if remove_nonlatin:
        text = re_sub(
            r"[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]", "", text)

    if remove_nonalphanumeric:
        text = re_sub(r'([^\s\w\']|_)+', " ", text)

    if use_number_special_token:
        text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>", text)
    elif remove_numbers:
        text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "", text)

    if deal_repetition:
        text = re_sub(r"([!?.]){2,}", r"\1", text)
        text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2", text)

    # remove multiple spaces
    text = re_sub(r"[ \s\t\n]+", " ", text)

    # segment and join words
    # e.g.427886834527
    # text = " ".join(ws.segment(text))

    if preserve_case:
        return text
    return text.strip().lower()


if __name__ == '__main__':
    # nltk.download('stopwords')
    ws.load()

    df = pd.read_csv('data/raw/train.csv', keep_default_na=False)
    # df2 = pd.read_csv('data/raw/test.csv', keep_default_na=False)

    for row in df.itertuples():
        df.set_value(row.Index, 'comment_text', clean_text(row.comment_text,
                                                           remove_nonalphanumeric=True,
                                                           use_number_special_token=True,
                                                           remove_numbers=False,
                                                           deal_repetition=True,
                                                           preserve_case=False,
                                                           remove_hashtag_at_end=False,
                                                           separate_contractions=True,
                                                           separate_punctuations=False,
                                                           remove_url=True,
                                                           remove_stopwords=False,
                                                           remove_nonlatin=True,
                                                           row_id=row.id))
        # break

    # for row in df2.itertuples():
    #     df2.set_value(row.Index, 'comment_text', clean_text(row.comment_text,
    #                                                         remove_nonalphanumeric=True,
    #                                                         use_number_special_token=True,
    #                                                         remove_numbers=False,
    #                                                         deal_repetition=True,
    #                                                         preserve_case=False,
    #                                                         remove_hashtag_at_end=False,
    #                                                         separate_contractions=True,
    #                                                         separate_punctuations=False,
    #                                                         remove_url=True,
    #                                                         remove_stopwords=False))

    df.to_csv('data/preprocessed/train_cleaned.csv', index=False)
    # df2.to_csv('data/preprocessed/test_cleaned.csv', index=False)
