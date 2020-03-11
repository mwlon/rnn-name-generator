import torch

START_CHAR = '^'
END_CHAR = '\n'

class Charset:
  def __init__(self, chars):
    self.chars = chars
    self.n_char = len(chars)

    self.char_to_idx = {}
    for i in range(self.n_char):
      self.char_to_idx[chars[i]] = i
    self.char_to_idx[START_CHAR] = self.n_char

  def index(self, char):
    return self.char_to_idx[char]

  def char_to_tens(self, char):
    tensor = torch.zeros(1, self.n_char + 1)
    tensor[0][self.char_to_idx[char]] = 1
    return tensor

  #turn a l-character name into l x 1 x n_char one-hot tensor
  def name_to_batch(self, name):
    l = len(name)
    tensor = torch.zeros(l, 1 , self.n_char + 1)
    for i in range(l):
      tensor[i][0][self.char_to_idx[name[i]]] = 1
    return tensor

letters = [chr(i) for i in range(97, 123)]
numbers = [str(i) for i in range(10)]

default_charsets = {
  'pause_alpha_num': Charset(['.', '-', ' ', END_CHAR] + letters + numbers),
  'connect_alpha': Charset(['-', ' ', END_CHAR] + letters)
}
