import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional
import torch.nn as nn
from collections import Counter
from charset import START_CHAR, END_CHAR


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

  #some utils for generating a name from the model
  def generate_name(self, charset):
    char = START_CHAR
    name = ''
    hidden = self.init_hidden()
    while char != END_CHAR:
      inp = Variable(charset.char_to_tens(char))
      output, hidden = self.forward(inp, hidden, is_training=False)
      p = output.data.numpy()[0]
      char = np.random.choice(charset.chars, p=p)
      name += char
    return name
  
  def generate_name_greedy(self, charset):
    char = START_CHAR
    name = ''
    hidden = self.init_hidden()
    while char != END_CHAR:
      inp = Variable(charset.char_to_tens(char))
      output, hidden = self.forward(inp, hidden, is_training=False)
      p = output.data.numpy()[0]
      i = np.argmax(p)
      char = charset.chars[i]
      name += char
    return name
  
  def generate_name_optimal(self, charset, search_p=0.001, omit_english=False):
    if omit_english:
      english_words = set(open('/usr/share/dict/words').read().split('\n'))
    complete_words = []
    frontier = [(START_CHAR, 1.0)]
    i = 0
    while i < len(frontier):
      seq, c_p = frontier[i]
      #print i, seq, c_p
      hidden = self.init_hidden()
      for char in seq:
        inp = Variable(charset.char_to_tens(char))
        output, hidden = self.forward(inp, hidden, is_training=False)
      p = output.data.numpy()[0]
      for j in range(charset.n_char):
        new_seq_p = c_p * p[j]
        if new_seq_p > search_p:
          new_char = charset.chars[j]
          y = (seq + new_char, new_seq_p)
          if new_char == END_CHAR:
            if not omit_english or y[0].rstrip(END_CHAR).lstrip(START_CHAR) not in english_words:
              complete_words.append(y)
          else:
            frontier.append(y)
      i += 1
    return complete_words
