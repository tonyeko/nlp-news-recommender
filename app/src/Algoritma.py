# some_file.py

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, '../../topic_classification')
from main import main_classification

def mainProgram(berita):

    hasil = []
    hasil.append(berita)
    topic = main_classification(hasil)
    return topic
