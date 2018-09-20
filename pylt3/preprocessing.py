from pathlib import Path
import spacy


def tokenize(in_src, out, lang, convert_html=False, convert_unicode=False, keep_empty_lines=False, verbose=False):
    src_path = Path(in_src).resolve()
    out_path = Path(out).resolve()

    if convert_html:
        from html import unescape

    # Only tokenize, don't execute the rest of the pipeline
    nlp = spacy.load(lang, disable=['parser', 'ner', 'tagger', 'textcat'])

    with open(str(src_path), 'r', encoding='utf-8') as fh_in, \
            open(str(out_path), 'w', encoding='utf-8') as fh_out:

        lines_removed = 0
        for line_idx, src_line in enumerate(fh_in):
            if verbose:
                print(f'Processing l. {line_idx+1}...', end="\r", flush=True)

            if convert_html:
                src_line = unescape(src_line)

            if convert_unicode:
                src_line = src_line.encode('utf-8').decode('unicode-escape')

            src_doc = nlp(src_line)
            src_line = ' '.join(str(token) for token in src_doc)

            src_line = src_line.strip()

            if keep_empty_lines or src_line != '':
                fh_out.write(src_line + '\n')
            else:
                lines_removed += 1

    if verbose:
        end_credits = 'Done processing:'
        end_credits += '\n\t- tokenized;'

        if lines_removed > 0:
            end_credits += f'\n\t- removed {lines_removed} empty lines;'

        end_credits += f'\n\t- wrote {line_idx - lines_removed} lines in total.'

        end_credits += f'\n\nResults in {str(out_path)}'

        print(end_credits)
