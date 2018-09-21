from pathlib import Path
import re


def tokenize_file(src, out, tokenizer=None, lang=None, html=False, unicode=False, lowercase=False, verbose=False):
    unescape = None

    src_path = Path(src).resolve()
    out_path = Path(out).resolve()

    if html:
        from html import unescape

    if tokenizer is None or tokenizer == 'spacy':
        if lang is None:
            raise TypeError('When using spaCy, you must specify the source and target languages. If you want to'
                            ' solely rely on NLTK, specify nltk as the tokenizer')

    # Set up tokenizer
    spacy_tok, nltk_tok = _set_tokenizer(tokenizer, lang, verbose)

    line_idx = 0
    lines_removed = 0
    with open(str(src_path), 'r', encoding='utf-8') as fh_in, \
            open(str(out_path), 'w', encoding='utf-8') as fh_out:

        for line_idx, line in enumerate(fh_in):
            if verbose:
                print(f'Processing l. {line_idx+1}...', end="\r", flush=True)

            if html:
                line = unescape(line)

            if unicode:
                line = unicode_replace(line)

            if spacy_tok:
                doc = spacy_tok(line)
                line = ' '.join(str(token) for token in doc)
            elif nltk_tok:
                line = ' '.join(nltk_tok(line))

            line = line.strip()

            if lowercase:
                line = line.lower()

            if line != '':
                fh_out.write(line + '\n')
            else:
                lines_removed += 1

    if verbose:
        _print_tokenize_summary(out_path, line_idx, lines_removed, html, unicode, lowercase)


def tokenize_string(line, tokenizer=None, lang=None, html=False, unicode=False, lowercase=False, verbose=False):
    unescape = None

    if html:
        from html import unescape

    if tokenizer is None or tokenizer == 'spacy':
        if lang is None:
            raise TypeError('When using spaCy, you must specify the source and target languages. If you want to'
                            ' solely rely on NLTK, specify nltk as the tokenizer')

    # Set up tokenizer
    spacy_tok, nltk_tok = _set_tokenizer(tokenizer, lang, verbose)

    if html:
        line = unescape(line)

    if unicode:
        line = unicode_replace(line)

    if spacy_tok:
        doc = spacy_tok(line)
        line = ' '.join(str(token) for token in doc)
    elif nltk_tok:
        line = ' '.join(nltk_tok(line))

    line = line.strip()

    if lowercase:
        line = line.lower()

    return line


def unicode_replace(s):
    def repl(match):

        match = match.group()
        try:
            return match.encode('utf-8').decode('unicode-escape')
        except UnicodeDecodeError:
            return match

    return re.sub(r'(?<!\b[a-zA-Z]:)(\\u[0-9A-Fa-f]{4})', repl, s)


def _get_spacy(lang):
    import spacy
    nlp = spacy.load(lang, disable=['parser', 'ner', 'tagger', 'textcat'])

    return nlp


def _print_tokenize_summary(out_path, line_idx, lines_removed, html, unicode, lowercase):
    end_credits = 'Done tokenizing:'
    if html:
        end_credits += '\n\t- unescaped HTML entities;'
    if unicode:
        end_credits += '\n\t- replaced unicode sequences;'
    if lowercase:
        end_credits += '\n\t- lower-cased characters;'
    if lines_removed > 0:
        end_credits += f'\n\t- removed {lines_removed} empty lines;'

    end_credits += f'\n\t- wrote {line_idx - lines_removed} lines in total.'

    end_credits += f'\n\nResults in {str(out_path)}'

    print(end_credits)


def _set_tokenizer(tokenizer, lang, verbose):
    spacy_nlp = None
    word_tokenize = None

    try:
        if tokenizer is None:
            try:
                if verbose:
                    print('Using spaCy\'s tokenizer...', flush=True)
                spacy_nlp = _get_spacy(lang)
            except (ModuleNotFoundError, ImportError, AttributeError, OSError) as e:
                print(f'Failed to load spaCy: {e}...\nWill try to use NLTK instead', flush=True)
                from nltk import word_tokenize
        else:
            if tokenizer == 'spacy':
                if verbose:
                    print('Using spaCy\'s tokenizer...', flush=True)
                    spacy_nlp = _get_spacy(lang)
            elif tokenizer == 'nltk':
                if verbose:
                    print('Using NLTK\'s tokenizer...', flush=True)
                from nltk import word_tokenize
            else:
                raise ValueError('tokenizer has to be None (default), spacy, or nltk')
    except (ModuleNotFoundError, ImportError) as e:
        error = 'To tokenize your input, you need the spaCy library or NLTK installed.'
        if tokenizer is not None:
            error += f' You chose to tokenize with {tokenizer}, which is not installed.'
        error += f'\nError message: {e}'
        raise type(e)(error)
    except (OSError, AttributeError) as e:
        raise type(e)(f'Something went wrong while loading a tokenizer.\nError message: {e}')

    return spacy_nlp, word_tokenize
