import requests
import re
import time
import sys

startup_name_regex = r'title=\\"([^\\]*)\\" data-type=\\"Startup\\"'
id_endpoint = 'https://angel.co/company_filters/search_data/'
name_endpoint = 'https://angel.co/companies/startups'

out = open('raw_names.txt', 'a+')

def add_query_results(opts={}):
  id_params = {
    'filter_data[company_types][]': 'Startup',
    'sort': 'signal'
  }
  name_params = {
    'sort': 'signal',
    'new': False
  }
  id_params.update(opts)
  
  ids = None
  content = None
  page_count = 1
  results_count = 0
  while ids is None or len(ids) > 0:
    id_params['page'] = page_count
    id_res = requests.get(id_endpoint, params=id_params)
    id_json = id_res.json()
  
    if 'ids' not in id_json:
      print 'WARNING: Too many ids found in this query! You might find more startups by using more granular queries.'
      break
  
    ids = id_json['ids']

    hex = id_json['hexdigest']
    total = id_json['total']
    name_params['ids[]'] = ids
    name_params['page'] = page_count
    name_params['hexdigest'] = hex
    name_params['total'] = total
    name_res = requests.get(name_endpoint, params=name_params)
    text = name_res.text
    matches = re.findall(startup_name_regex, text)
    for match in matches:
      print match
      out.write((match + '\n').encode('utf8'))
      results_count += 1
    print '\tprocessed page {} with {} results for this query so far'.format(page_count, results_count)
    time.sleep(0.5)
    page_count += 1
  return {'page_count': page_count, 'results_count': results_count}

def main():
  #the trick is to find queries that cover most of the space of startups
  queries = []
  #these first ones cover the startups that haven't raised any money
  #these are the ones we can't get all of, so warnings are fine
  low = 0
  high = 0.9
  while low < 10:
    for sort_col in ['signal', 'joined']:
      queries.append({
        'filter_data[raised][min]': 0,
        'filter_data[raised][max]': 0,
        'filter_data[signal][min]': low,
        'filter_data[signal][max]': high,
        'sort': sort_col
      })
    for market in ['E-Commerce', 'Education', 'Enterprise Software', 'Games', 'Healthcare', 'Mobile']:
      queries.append({
        'filter_data[raised][min]': 0,
        'filter_data[raised][max]': 0,
        'filter_data[signal][min]': low,
        'filter_data[signal][max]': high,
        'filter_data[markets][]': market
      })
    for tech in ['Javascript', 'Java', 'Python', 'HTML5', 'CSS', 'Machine Learning', 'Scala', 'Ruby On Rails', 'Node.js', 'AngularJS']:
      queries.append({
        'filter_data[raised][min]': 0,
        'filter_data[raised][max]': 0,
        'filter_data[signal][min]': low,
        'filter_data[signal][max]': high,
        'filter_data[teches][]': tech
      })
    low = high
    high += 1

  #these next ones cover ones that raised 1 - 200M
  low = 1
  high = 1000
  multiplier = 1.2
  cutoff = 2 * 10 ** 8
  while low < cutoff:
    queries.append(
      {'filter_data[raised][min]': low, 'filter_data[raised][max]': high}
    )
    low = high
    high = min(int(high * multiplier), cutoff)

  #this last one for the few companies with 200M+ raised
  queries.append({'filter_data[raised][min]': cutoff, 'filter_data[raised][max]': 10**11})

  for i in range(len(queries)):
    query = queries[i]
    print 'starting query {} with params {}'.format(i, query)
    add_query_results(query)

if __name__ == "__main__":
  main()
