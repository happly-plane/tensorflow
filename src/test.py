import tensorflow as tf

import numpy as np
import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"




# Read, then decode for py2 compat.
text = open(r"dataset\xiaoshuo.txt", 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print(f'Length of text: {len(text)} characters')

# Take a look at the first 250 characters in text
print(text[:250])

# The unique characters in the file
vocab = sorted(set(text))
print(f'{len(vocab)} unique characters')




example_texts = ['abcdefg', 'xyz']

chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')
print(chars)
ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)
ids = ids_from_chars(chars)
print(ids)