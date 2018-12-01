import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional
import torch.nn as nn
import math
from model import *
import json

epochs = 35
base_lr = 0.008
min_lr_factor = 0.005
tracking_filename = 'tracking.json'

train_names = []
val_names = []
for line in open('train.txt', 'r'):
  assert line[-1] == '\n'
  train_names.append(line)
for line in open('val.txt', 'r'):
  assert line[-1] == '\n'
  val_names.append(line)
n_train = len(train_names)
n_val = len(val_names)

criterion = nn.NLLLoss()

def train_batch(name, lr):
  rnn.zero_grad()
  stuff = eval_batch(name, is_training=True)
  loss = stuff['ll']
  loss.backward()
  for p in rnn.parameters():
    p.data.add_(-lr, p.grad.data)

  return loss.data[0], stuff['correct_chars']

def eval_batch(name, is_training=False):
  l = len(name)
  tens = Variable(name_to_batch(name))
  hidden = rnn.init_hidden()
  correct_names = 1
  correct_chars = 0
  ll = 0

  for i in range(l):
    inp = tens[i - 1] if i > 0 else Variable(char_to_tens(start_char))
    output, hidden = rnn(inp, hidden, is_training=is_training)
    logp = torch.log(output + 10**-7)
    values, indices = torch.max(output, 1)
    next_char_idx = char_to_idx[name[i]]
    target = Variable(torch.LongTensor([next_char_idx]))
    ll += criterion(logp, target)

    if next_char_idx == indices.data.numpy()[0]:
      correct_chars += 1
    else:
      correct_names = 0
  
  return {
    'correct_names': correct_names,
    'correct_chars': correct_chars,
    'total_names': 1,
    'total_chars': l,
    'll': ll
  }

def summarize_metrics(metrics):
  res = {}
  for metric in metrics:
    for k, v in metric.iteritems():
      if k not in res:
        res[k] = 0.0
      res[k] += v
  res['ll'] = res['ll'].data[0]
  ll = res['ll'] / res['total_names']
  ll_per_char = res['ll'] / res['total_chars']
  char_acc = res['correct_chars'] / res['total_chars']
  name_acc = res['correct_names'] / res['total_names']
  res['eval_accuracy'] = char_acc
  res['eval_loss'] = ll_per_char
  print 'EVAL CHAR ACCURACY:\t{}'.format(char_acc)
  print 'EVAL WORD ACCURACY:\t{}'.format(name_acc)
  print 'EVAL LIKELIHOOD:\t{}'.format(ll)
  print 'EVAL PER-CHAR LIKELIHOOD:\t{}'.format(ll_per_char)
  return res

tracking = []
for i in range(epochs):
  np.random.shuffle(train_names)
  theta = np.pi * i / float(epochs - 1)
  lr = base_lr * (min_lr_factor + (1.0 - min_lr_factor) * (1.0 + np.cos(theta)) / 2.0)

  total_chars = 0
  total_loss = 0
  total_correct = 0
  for j in range(n_train):
    name = train_names[j]
    loss, correct = train_batch(name, lr)
    total_chars += len(name)
    total_correct += correct
    total_loss += loss
    if math.isnan(loss):
      print "OH NOES, LOSS IS NAN AROUND {}".format(train_names[j - 1: j + 2])
    if j % 100 == 0:
      print i, j, name.rstrip('\n'), loss / len(name)
      print 'generated name: {}'.format(generate_name())
  train_loss = total_loss / total_chars
  train_accuracy = total_correct / float(total_chars)
  print 'TRAIN CHAR ACCURACY {}'.format(train_accuracy)
  print 'TRAIN CHAR LOSS {}'.format(train_loss)

  metrics = []
  for j in range(n_val):
    name = val_names[j]
    metrics.append(eval_batch(name))
  summary = summarize_metrics(metrics)
  summary['epoch'] = i + 1
  summary['lr'] = lr
  summary['train_loss'] = train_loss
  summary['train_accuracy'] = train_accuracy

  tracking.append(summary)
  out = open(tracking_filename, 'w')
  json.dump(tracking, out)
  out.close()
  torch.save(rnn.state_dict(), 'checkpoint.pth')
