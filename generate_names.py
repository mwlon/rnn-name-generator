from model import *
import torch

ckpt = torch.load('checkpoint.pth')
rnn.load_state_dict(ckpt)

#make a list of names
#out = open('generated_names.txt', 'w')
#for i in range(20000):
#  name = generate_name()
#  out.write(name)

#find what the model would generate if choosing characters greedily
#print generate_name_greedy()

#or find the most likely names
#print generate_name_optimal(0.00003)
