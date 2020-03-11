from model import Rnn
from charset import default_charsets, START_CHAR, END_CHAR
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
  '--checkpoint_path',
  type=str,
  default='checkpoint.pth',
  help='checkpoint file to use',
)
parser.add_argument(
  '--charset',
  type=str,
  default='pause_alpha_num',
  help='which pre-defined charset to use. see charset.py for the map',
)
parser.add_argument(
  '--hidden_sizes',
  type=str,
  default='150,150,150',
  help='sizes of the intermediate hidden layers',
)
parser.add_argument(
  '--mode',
  type=str,
  default='random',
  help='whether to generate random, greedy, or optimal names'
)
parser.add_argument(
  '--num_names',
  type=int,
  default=10000,
  help='number of random names to generate, if applicable'
)
parser.add_argument(
  '--name_prob',
  type=float,
  default=3E-5,
  help='minimum likelihood of names to generate, if searching optimal'
)
parser.add_argument(
  '--out_file',
  type=str,
  default=None,
  help='file to write to, if desired'
)
parser.add_argument(
  '--print_out',
  dest='print_out',
  action='store_const',
  const=True,
  default=False,
  help='whether to print results'
)
args = parser.parse_args()

charset = default_charsets[args.charset]
hidden_sizes = []
for size in args.hidden_sizes.split(','):
  hidden_sizes.append(int(size))
rnn = Rnn([charset.n_char + 1] + hidden_sizes + [charset.n_char])

ckpt = torch.load('checkpoint.pth')
rnn.load_state_dict(ckpt)

names = []

if args.mode == 'random':
  for i in range(args.num_names):
    names.append(rnn.generate_name(charset))
elif args.mode == 'greedy':
  for i in range(args.num_names):
    names.append(rnn.generate_greedy(charset))
elif args.mode == 'optimal':
  optimal = rnn.generate_name_optimal(charset, args.name_prob)
  for name, prob in sorted(optimal, key=lambda x: -x[1]):
    names.append(name)

names = [name.lstrip(START_CHAR).rstrip(END_CHAR) for name in names]

if args.print_out:
  for name in names:
    print(name)

if args.out_file:
  out = open(args.out_file, 'w')
  for name in names:
    out.write(name + '\n')
