import os
import subprocess
import tempfile


class PTBTokenizer(object):
    """Python wrapper of Stanford PTBTokenizer"""

    corenlp_jar = "stanford-corenlp-3.4.1.jar"
    punctuations = [
        "''",
        "'",
        "``",
        "`",
        "-LRB-",
        "-RRB-",
        "-LCB-",
        "-RCB-",
        ".",
        "?",
        "!",
        ",",
        ":",
        "-",
        "--",
        "...",
        ";",
    ]
    eos_token = "<eos>"

    @classmethod
    def tokenize(cls, corpus, add_eos=False):
        cmd = [
            "java",
            "-cp",
            cls.corenlp_jar,
            "edu.stanford.nlp.process.PTBTokenizer",
            "-preserveLines",
            "-lowerCase",
        ]

        if isinstance(corpus, list) or isinstance(corpus, tuple):
            if isinstance(corpus[0], list) or isinstance(corpus[0], tuple):
                corpus = {i: c for i, c in enumerate(corpus)}
            else:
                corpus = {
                    i: [
                        c,
                    ]
                    for i, c in enumerate(corpus)
                }

        # prepare data for PTB Tokenizer
        tokenized_corpus = {}
        image_id = [k for k, v in list(corpus.items()) for _ in range(len(v))]
        sentences = "\n".join(
            [c.replace("\n", " ") for k, v in corpus.items() for c in v]
        )

        # save sentences to temporary file
        path_to_jar_dirname = os.path.dirname(os.path.abspath(__file__))
        tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=path_to_jar_dirname)
        tmp_file.write(sentences.encode())
        tmp_file.close()

        # tokenize sentence
        cmd.append(os.path.basename(tmp_file.name))
        p_tokenizer = subprocess.Popen(
            cmd,
            cwd=path_to_jar_dirname,
            stdout=subprocess.PIPE,
            stderr=open(os.devnull, "w"),
        )
        token_lines = p_tokenizer.communicate(input=sentences.rstrip())[0]
        token_lines = token_lines.decode()
        lines = token_lines.split("\n")
        # remove temp file
        os.remove(tmp_file.name)

        # create dictionary for tokenized captions
        for k, line in zip(image_id, lines):
            if k not in tokenized_corpus:
                tokenized_corpus[k] = []
            tokenized_caption = " ".join(
                [w for w in line.rstrip().split(" ") if w not in cls.punctuations]
            )
            if add_eos:
                tokenized_caption += " {}".format(cls.eos_token)
            tokenized_corpus[k].append(tokenized_caption)

        return tokenized_corpus
