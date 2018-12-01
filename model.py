import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional
import torch.nn as nn
import math
from collections import Counter

#generate simple mapping from index -> char and vice versa
chars = ['.', '-', ' ', '\n']
for i in range(26):
  chars.append(chr(i + 97))
for i in range(10):
  chars.append(str(i))
print 'running with {} chars: {}'.format(len(chars), chars)
n_char = len(chars)
start_char = '^'

char_to_idx = {}
for i in range(n_char):
  char_to_idx[chars[i]] = i
char_to_idx[start_char] = n_char

def char_to_tens(char):
  tensor = torch.zeros(1, n_char + 1)
  tensor[0][char_to_idx[char]] = 1
  return tensor

#turn a l-character name into l x 1 x n_char one-hot tensor
def name_to_batch(name):
  l = len(name)
  tensor = torch.zeros(l, 1 , n_char + 1)
  for i in range(l):
    tensor[i][0][char_to_idx[name[i]]] = 1
  return tensor

class Rnn(nn.Module):
  def __init__(self, sizes):
    super(Rnn, self).__init__()
    self.sizes = sizes
    self.n_layer = len(sizes) - 1

    self.layers = []
    for i in range(self.n_layer):
      o = nn.Linear(sizes[i] + sizes[i + 1], sizes[i + 1])
      setattr(self, 'o{}'.format(i), o)
      self.layers.append(o)

    self.softmax = nn.Softmax(dim=1)

  def forward(self, inp, hiddens, is_training):
    combined = torch.cat((inp, hiddens[0]), 1)
    new_hiddens = []
    for i in range(self.n_layer):
      output = self.layers[i](combined)
      if i < self.n_layer - 1:
        output = functional.relu(output)
        output = functional.dropout(output, training=is_training, p=0.05)
        combined = torch.cat((output, hiddens[i + 1]), 1)
      new_hiddens.append(output)
    output = self.softmax(output) + 10**-7
    return output, new_hiddens

  def init_hidden(self):
    return [Variable(torch.zeros(1, self.sizes[i + 1])) for i in range(self.n_layer)]

#specify depth and width of RNN layers here
#input  is n_char + 1 for characters (including stop char and start char)
#output is n_char for characters (including stop char)
rnn = Rnn([n_char + 1, 150, 150, 150, n_char])

#some utils for generating a name from the model
def generate_name():
  char = '^'
  name = ''
  hidden = rnn.init_hidden()
  while char != '\n':
    inp = Variable(char_to_tens(char))
    output, hidden = rnn.forward(inp, hidden, is_training=False)
    p = output.data.numpy()[0]
    char = np.random.choice(chars, p=p)
    name += char
  return name

def generate_name_greedy():
  char = '^'
  name = ''
  hidden = rnn.init_hidden()
  while char != '\n':
    inp = Variable(char_to_tens(char))
    output, hidden = rnn.forward(inp, hidden, is_training=False)
    p = output.data.numpy()[0]
    i = np.argmax(p)
    char = chars[i]
    name += char
  return name

def generate_name_optimal(search_p=0.001):
  english_words = set(open('/usr/share/dict/words').read().split('\n'))
  complete_words = []
  frontier = [('^', 1.0)]
  i = 0
  while i < len(frontier):
    seq, c_p = frontier[i]
    #print i, seq, c_p
    hidden = rnn.init_hidden()
    for char in seq:
      inp = Variable(char_to_tens(char))
      output, hidden = rnn.forward(inp, hidden, is_training=False)
    p = output.data.numpy()[0]
    for j in range(n_char):
      new_seq_p = c_p * p[j]
      if new_seq_p > search_p:
        new_char = chars[j]
        y = (seq + new_char, new_seq_p)
        if new_char == '\n':
          if y[0].rstrip('\n').lstrip('^') not in english_words:
            complete_words.append(y)
        else:
          frontier.append(y)
    i += 1
  return complete_words
