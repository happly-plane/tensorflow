import tensorflow as tf
from tensorflow import keras
import numpy as np
import re

text = open('dataset/xiaoshuo.txt', "rb").read().decode(encoding= 'utf-8')
text = text.replace(" ", "").replace("\n","").replace("\r","").replace("\u3000","")

print('length of text: {}'.format(len(text)))                                                           

print(text[:250])

vocab = sorted(set(text))

print('{} unique characters'.format(len(vocab)))


char2idx = {u:i for i,u in enumerate(vocab)}

idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])


print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')


# 显示文本首 13 个字符的整数映射
print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

# 设定每个输入句子长度的最大值
seq_length = 100
examples_per_epoch = len(text)//seq_length

# 创建训练样本 / 目标
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
  print(idx2char[i.numpy()])
  sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):
  print(repr(''.join(idx2char[item.numpy()])))