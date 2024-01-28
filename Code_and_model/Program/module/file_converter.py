import os
import pandas as pd
import numpy as np
import codecs


def _to_utf8(filename: str, encoding="latin1", blocksize=1048576):
    tmpfilename = filename + ".tmp"
    with codecs.open(filename, "r", encoding) as source:
        with codecs.open(tmpfilename, "w", "utf-8") as target:
            while True:
                contents = source.read(blocksize)
                if not contents:
                    break
                target.write(contents)
    os.rename(filename, filename + '_old')
    # replace the original file
    os.rename(tmpfilename, filename)