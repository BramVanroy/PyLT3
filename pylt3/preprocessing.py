from pathlib import Path
import re
import locale
from html import unescape
from linecache import getline

from pylt3.utils.type_helpers import verify_kwargs

try:
    from nltk import word_tokenize
except ModuleNotFoundError:
    pass


class PreProcessor:
    def __init__(self, tokenizer=None, lang=None):
        self.digit_table = str.maketrans('0123456789', '1111111111')
        self.url_regex = r'((([A-Za-z]{3,9}:(?:\/\/)?)(?:[-;:&=\+\$,\w]+@)?[A-Za-z0-9.-]+|(?:www\d*\.|[-;:&=\+\$,\w]+@)[A-Za-z0-9.,)(-]+)((?:\/[\+~%\/.\w_-]*)?\??(?:[-\+=&;%@.\w_]*)#?(?:[\w]*))?)'
        self.unicode_regex = r'(?<!\b[a-zA-Z]:)(\\u[0-9A-Fa-f]{4})'
        self.punct_regex = r'(?<!\w)[^\sa-zA-Z0-9_]+(?!\w)'
        self.spacy = None
        self.tagmap = None
        self.tokenizer = tokenizer
        self.lang = lang

        if tokenizer is not None:
            self.set_tokenizer(tokenizer, lang, verbose=True)

    def set_tokenizer(self, tokenizer, lang, verbose):
        try:
            if tokenizer is None:
                try:
                    if verbose:
                        print('Using spaCy\'s tokenizer...', flush=True)
                    self.spacy = self._get_spacy(lang)
                    self.tagmap = self.spacy.Defaults.tag_map
                    self.tokenizer = 'spacy'
                except (ModuleNotFoundError, ImportError, AttributeError, OSError) as e:
                    if verbose:
                        print(f'Failed to load spaCy: {e}...\nWill try to use NLTK instead', flush=True)
                    try:
                        self.tokenizer = 'nltk'
                    except (ModuleNotFoundError, ImportError, AttributeError, OSError) as e:
                        if verbose:
                            print(f'Failed to load nltk: {e}...\nWill use naive tokenizer instead', flush=True)
                        self.tokenizer = 'naive'
            else:
                if tokenizer == 'spacy':
                    if verbose:
                        print('Using spaCy\'s tokenizer...', flush=True)
                        self.spacy = self._get_spacy(lang)
                        self.tagmap = self.spacy.Defaults.tag_map
                        self.tokenizer = 'spacy'
                elif tokenizer == 'nltk':
                    if verbose:
                        print('Using NLTK\'s tokenizer...', flush=True)
                    self.tokenizer = 'nltk'
                elif tokenizer == 'naive':
                    if verbose:
                        print('Using naive tokenizer...', flush=True)
                    self.tokenizer = 'naive'
                else:
                    raise ValueError('tokenizer has to be None (default), spacy, nltk, or naive')
        except (ModuleNotFoundError, ImportError) as e:
            error = 'To tokenize your input, you need the spaCy library or NLTK installed.'
            if tokenizer is not None:
                error += f' You chose to tokenize with {tokenizer}, which is not installed.'
            error += f'\nError message: {e}'
            raise type(e)(error)
        except (OSError, AttributeError) as e:
            raise type(e)(f'Something went wrong while loading a tokenizer.\nError message: {e}')

    def normalize_digits_string(self, line):
        return line.translate(self.digit_table)

    def normalize_digits_file(self, src, out, **kwargs):
        default_params = {'src_encoding': locale.getpreferredencoding(), 'out_encoding': locale.getpreferredencoding()}
        kwargs = verify_kwargs(default_params, kwargs)

        src_path = Path(src).resolve()
        out_path = Path(out).resolve()

        with open(str(src_path), 'r', encoding=kwargs['src_encoding']) as fhin, \
                open(str(out_path), 'w', encoding=kwargs['out_encoding']) as fhout:

            for line in fhin:
                fhout.write(line.translate(self.digit_table))

    def normalize_url_file(self, src, out, repl='@url@', **kwargs):
        default_params = {'src_encoding': locale.getpreferredencoding(), 'out_encoding': locale.getpreferredencoding()}
        kwargs = verify_kwargs(default_params, kwargs)

        src_path = Path(src).resolve()
        out_path = Path(out).resolve()

        with open(str(src_path), 'r', encoding=kwargs['src_encoding']) as fhin, \
                open(str(out_path), 'w', encoding=kwargs['out_encoding']) as fhout:

            for line in fhin:
                fhout.write(re.sub(self.url_regex, repl, line))

    def normalize_url_string(self, line, repl='@url@'):
        return re.sub(self.url_regex, repl, line)

    def tokenize_file(self, src, out, html=False, unicode=False, keep_empty=True, lowercase=False, verbose=False):
        src_path = Path(src).resolve()
        out_path = Path(out).resolve()

        line_idx = 0
        empty_lines = 0
        with open(str(src_path), 'r', encoding='utf-8') as fh_in, \
                open(str(out_path), 'w', encoding='utf-8') as fh_out:

            for line_idx, line in enumerate(fh_in):
                if verbose:
                    print(f'Processing l. {line_idx+1}...', end="\r", flush=True)

                line = self.tokenize_string(line, html, unicode, lowercase)

                if line != '':
                    fh_out.write(line + '\n')
                else:
                    if keep_empty:
                        fh_out.write('\n')
                    empty_lines += 1

        if verbose:
            PreProcessor._print_file_tokenize_summary(out_path, line_idx, empty_lines, html, unicode, keep_empty, lowercase)

    def tokenize_string(self, line, html=False, unicode=False, lowercase=False):
        if html:
            line = unescape(line)

        if unicode:
            line = self.unicode_replace(line)

        if self.tokenizer == 'spacy':
            doc = self.spacy(line)
            line = ' '.join(str(token) for token in doc)
        elif self.tokenizer == 'nltk':
            line = ' '.join(word_tokenize(line))
        elif self.tokenizer == 'naive':
            line = re.sub(r"\b", " ", line)
            line = ' '.join(line.split())

        line = line.strip()

        if lowercase:
            line = line.lower()

        return line

    def unicode_replace(self, line):
        def repl(match):
            match = match.group()
            try:
                return match.encode('utf-8').decode('unicode-escape')
            except UnicodeDecodeError:
                return match

        return re.sub(self.unicode_regex, repl, line)

    def remove_empty_lines(self, files_in, files_out, min_word_count=0, remove_everywhere=True):
        if len(files_in) != len(files_out):
            raise ValueError("The number of input and output files has to be equal.")

        paths_in = [Path(f).resolve() for f in files_in]
        paths_out = [Path(f).resolve() for f in files_out]
        fhs_out = [open(p, 'w', encoding='utf-8') for p in paths_out]

        def _get_word_count(text):
            # Find number of punctuation-only tokens, subtract of total token size
            line_len = len(list(filter(None, text.split(' '))))
            nro_punct = len(re.findall(self.punct_regex, text))

            return line_len - nro_punct

        nro_empty_lines = 0
        with open(paths_in[0], encoding='utf-8') as first_fhin:
            for line_no, line in enumerate(first_fhin, 1):
                lines = [line]
                lines.extend([getline(str(p), line_no) for p in paths_in[1:]])
                lines = list(map(str.strip, lines))

                if remove_everywhere and '' in lines:
                    print(f"Empty line found...")
                    nro_empty_lines += 1
                    continue

                # If there's a min_word_count, discard lines that have less than the required
                # number of tokens that not only consist out of punctuation
                if min_word_count:
                    nro_words = list(map(_get_word_count, lines))
                    less_than_min = (nr < min_word_count for nr in nro_words)

                    if remove_everywhere and any(less_than_min):
                        print(f"Line with too few words (less than {min_word_count}) found...")
                        nro_empty_lines += 1
                        continue

                for file_idx, file_line in enumerate(lines):
                        if file_line == '' or (min_word_count and (nro_words[file_idx] < min_word_count)):
                            continue

                        fhs_out[file_idx].write(f"{file_line}\n")

        _ = [fh.close() for fh in fhs_out]

        print(f"Finished processing files, removed {nro_empty_lines} lines.")

    def _get_morphology(self, tag):
        """ Taken from spacy_conll. Probably should import from there. """
        if not self.tagmap or tag not in self.tagmap:
            return '_'
        else:
            feats = [f'{prop}={val}' for prop, val in self.tagmap[tag].items() if not PreProcessor._is_number(prop)]
            if feats:
                return '|'.join(feats)
            else:
                return '_'

    @staticmethod
    def _get_spacy(lang):
        import spacy
        try:
            lang = 'en_core_web_sm' if lang == 'en' else lang
            nlp = spacy.load(lang, disable=['parser', 'ner', 'tagger', 'textcat'])
        except OSError:
            raise OSError(f"Something went wrong when trying to load spaCy's tokenizer."
                          f" You probably forgot to specify a 'lang' or the given 'lang' model is not installed.")
        return nlp

    @staticmethod
    def _print_file_tokenize_summary(out_path, line_idx, empty_lines, html, unicode, keep_empty, lowercase):
        end_credits = 'Done tokenizing:'
        if html:
            end_credits += '\n\t- unescaped HTML entities;'
        if unicode:
            end_credits += '\n\t- replaced unicode sequences;'
        if lowercase:
            end_credits += '\n\t- lower-cased characters;'

        if empty_lines > 0:
            if keep_empty:
                end_credits += f'\n\t- kept {empty_lines} empty lines;'
            else:
                end_credits += f'\n\t- removed {empty_lines} empty lines;'

        end_credits += f'\n\t- wrote {line_idx - empty_lines} lines in total.'

        end_credits += f'\n\nResults in {str(out_path)}'

        print(end_credits)

    @staticmethod
    def _is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
