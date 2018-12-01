import re
import io
import unicodedata

okchars = set([
  '-', '.', ' '
])
for i in range(10):
  okchars.add(str(i))
for i in range(26):
  okchars.add(chr(i + 97))

def only_ok(s):
  res = ''
  for ch in s:
    if ch in okchars:
      res += ch
  return res

def clean_ends(s):
  bad_start = set(['-', '.', ' '])
  bad_end = set(['-', ' '])
  start = len(s)
  end = 0
  for i in range(len(s)):
    char = s[i]
    if char not in bad_start:
      start = min(start, i)
    if char not in bad_end:
      end = i + 1
  return s[start:end]

def clean_multispace(s):
  res = ''
  in_space = False
  for char in s:
    if char == ' ':
      if not in_space:
        res += char
      in_space = True
    else:
      res += char
      in_space = False
  return res

out = open('cleaned_names.txt', mode='w')
observed = set()
def process(fname):
  inp = io.open(fname, mode='r', encoding='utf-8')
  for line in inp:
    outline = line\
      .replace('&quot;', '"')\
      .replace('&#39;', '\'')\
      .lower()
    outline = re.sub(r'\(.*\)', '', outline)
    outline = unicodedata.normalize('NFKD', outline).encode('ascii', 'ignore')
    outline = only_ok(outline)
    outline = clean_ends(outline)
    outline = clean_multispace(outline)
    if outline != '' and outline not in observed:
      print '{} -> {}'.format(line.rstrip('\n').encode('ascii', 'ignore'), outline)
      observed.add(outline)
      out.write('{}\n'.format(outline))

#include an extra_names file for some startups that were oddly missing
inps = ['raw_names.txt', 'extra_names.txt']
for fname in inps:
  process(fname)
out.close()
