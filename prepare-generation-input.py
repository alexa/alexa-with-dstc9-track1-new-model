import os
import nltk
import sys
import json

data_dir = sys.argv[1]
data_split = sys.argv[2]
ks_file = sys.argv[3]
out_dir = sys.argv[4]

logs = json.load(open(os.path.join(data_dir, data_split, 'logs.json')))
labels = json.load(open(os.path.join(data_dir, data_split, 'labels.json')))
knowledge = json.load(open(os.path.join(data_dir, 'knowledge.json')))

ks_res = json.load(open(ks_file))

assert len(ks_res) == len(labels)

with open(os.path.join(out_dir, 'test.source'), 'w') as out_source, open(os.path.join(out_dir, 'test.target'),
                                                                         'w') as out_target:
    for idx, label in enumerate(ks_res):
        if label['target']:
            answer = knowledge[label['knowledge'][0]['domain']][str(label['knowledge'][0]['entity_id'])]['docs'][
                str(label['knowledge'][0]['doc_id'])]['body']
            if idx < len(labels) - 1:
                out_source.write(answer.replace('\n', ' ') + '\n')
            else:
                out_source.write(answer.replace('\n', ' '))

            try:
                response = labels[idx]['response']
            except:
                response = ' '
                # print(labels[idx])
            else:
                response = nltk.sent_tokenize(response)
                response = ' '.join(response[:-1]) if len(response) > 1 else response[0]
            if idx < len(labels) - 1:
                out_target.write(response.replace('\n', ' ') + '\n')
            else:
                out_target.write(response.replace('\n', ' '))