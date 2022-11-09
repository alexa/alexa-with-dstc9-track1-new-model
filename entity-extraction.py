import os, json
import sys
import re
from difflib import SequenceMatcher as SM
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from collections import Counter

dataroot = sys.argv[1]
dataset = sys.argv[2]
ktd_pred_file = sys.argv[3]

with open('multiwoz21.json', "r", encoding='utf-8') as f:
    raw_config = json.load(f)
label_maps = raw_config['label_maps']


def fuzzy_extract(qs, ls, threshold):
    qs_length = len(qs.split())
    max_sim_val = 0
    max_sim_string = u""

    for ngram in ngrams(ls.split(), qs_length + int(.2 * qs_length)):
        ls_ngram = u" ".join(ngram)
        similarity = SM(None, ls_ngram, qs).ratio()
        if similarity > max_sim_val:
            max_sim_val = similarity
            max_sim_string = ls_ngram

    if max_sim_val > threshold:
        return max_sim_string, max_sim_val
    else:
        return None


def normalize_entity(entity):
    sep_symbols = [' - ', ', ', '/']
    for sep_symbol in sep_symbols:
        if sep_symbol in entity:
            entity = entity.split(sep_symbol)[0].strip()

    entity = entity.replace('guesthouse', '').replace('guest house', '').strip(string.punctuation).strip()

    # we can detect place names in these entities and remove it if they appear in the end
    place_names = ["fisherman's wharf", "san francisco", "san francisco downtown", "san francisco union square"]
    for place_name in place_names:
        if entity.endswith(place_name):
            entity = entity.rstrip(place_name)

    if "&" in entity:
        return [entity, entity.replace('&', 'and')]
    else:
        if bool(re.search(r'\d', entity)):
            d = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
                 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten',
                 11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen',
                 15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen',
                 19: 'nineteen', 20: 'twenty',
                 30: 'thirty', 40: 'forty', 50: 'fifty', 60: 'sixty',
                 70: 'seventy', 80: 'eighty', 90: 'ninety'}
            entity_ = ''
            for idx, char in enumerate(entity):
                if not char.isdigit():
                    entity_ += char
                else:
                    entity_ += d[int(char)]
                    if idx < len(entity) - 1 and entity[idx + 1] != ' ':
                        entity_ += ' '
            return [entity, entity_]
        else:
            return [entity]


def check_substring_exist(qs, ls):
    if len(qs.split()) > 1:
        if qs in ls:
            index = ls.rindex(qs)
            if index - 3 > -1 and ls[(index - 3):(index - 1)] == 'in':
                return False
            else:
                return True
    else:
        return qs in word_tokenize(ls)


def entity_matching(entity, log_text):
    if check_substring_exist(entity, log_text):
        return log_text.rindex(entity)

    if entity in label_maps:
        for variant in label_maps[entity]:
            if check_substring_exist(variant, log_text):
                return log_text.rindex(variant)

    entity_names = normalize_entity(entity)
    for entity_name in entity_names:
        if check_substring_exist(entity_name, log_text):
            return log_text.rindex(entity_name)

    for entity_name in entity_names:
        fuzzy_match_res = fuzzy_extract(entity_name, log_text, 0.95)
        if fuzzy_match_res is not None:
            return log_text.index(fuzzy_match_res[0])

    return None


# read data
logs_file = os.path.join(dataroot, dataset, 'logs.json')
knowledge_file = os.path.join(dataroot, 'knowledge.json')
with open(logs_file, 'r') as f:
    logs = json.load(f)
with open(knowledge_file, 'r') as f:
    knowledges = json.load(f)

labels_file = os.path.join(dataroot, dataset, 'labels.json')
with open(labels_file, 'r') as f:
    labels = json.load(f)

all_entity_names = []
for domain, domain_dict in knowledges.items():
    if domain in ['train', 'taxi']:
        continue
    for doc_id, docs in domain_dict.items():
        all_entity_names.append((domain, doc_id, docs['name'].lower()))

# read knowledge seeking turn detection
ktd_preds = json.load(open(ktd_pred_file))

all_matching_res = []
num_tot = 0
for idx_, (log, label, ktd_pred) in enumerate(zip(logs, labels, ktd_preds)):
    if ktd_pred['target']:
        log_text = ' '.join([utt['text'] for utt in log]).lower()
        matching_res_ls = set()
        for entity_tup in all_entity_names:
            entity_domain, entity_id, entity_name = entity_tup
            match_res = entity_matching(entity_name, log_text)
            if match_res is not None:
                matching_res_ls.add((entity_domain, entity_id, entity_name, match_res))
        matching_res_ls = sorted(list(matching_res_ls), key=lambda x: x[-1])

        # here I want to resolve this error: [('attraction', '100026', 'cable car - california street line', 269), ('attraction', '100028', 'cable car - powell/mason line', 269), ('attraction', '100027', 'cable car - powell/hyde line', 269)]
        buffer = []
        buffer_num = []
        for idx, res in enumerate(matching_res_ls):
            if ' - ' in res[-2]:
                buffer.append((idx, res))
                buffer_num.append(res[-1])
        if len(buffer) > 1:
            most_num = max(set(buffer_num), key=buffer_num.count)
            buffer = [item for item in buffer if item[-1][-1] == most_num]
            idx_to_remove = set([item[0] for item in buffer if item[-1][-2].split(' - ')[-1] not in log_text])
            if len(idx_to_remove) < len(matching_res_ls):
                matching_res_ls = [res for idx, res in enumerate(matching_res_ls) if idx not in idx_to_remove]

        # remove the place entities
        if len(matching_res_ls) > 1:
            location_indicators = ['at ', 'in ', 'at the ', 'in the ']
            idx_to_remove = []
            for idx, res in enumerate(matching_res_ls):
                first_utt = log[0]['text'].lower()
                if res[-2] in first_utt:
                    position = first_utt.index(res[-2])
                    if any([True if first_utt[position - len(indicator):position] == indicator else False for indicator
                            in location_indicators]):
                        idx_to_remove.append(idx)
            if idx_to_remove:
                matching_res_ls = [res for idx, res in enumerate(matching_res_ls) if idx not in idx_to_remove]

        try:
            gold_entity_name = knowledges[label['knowledge'][0]['domain']][str(label['knowledge'][0]['entity_id'])][
                'name'].lower()
        except:
            gold_entity_name = ''

        try:
            rank = list(map(lambda x: x[2], matching_res_ls))[::-1].index(gold_entity_name) + 1
        except:
            rank = -1

        try:
            all_matching_res.append((matching_res_ls, idx_, label['knowledge'][0]['domain'],
                                     str(label['knowledge'][0]['entity_id']), gold_entity_name, rank))
        except:
            all_matching_res.append((matching_res_ls, idx_,))

        # if rank != 1:
        #     print(idx_, num_tot, matching_res_ls, gold_entity_name, rank)
        #     print(log)
        #     print()

        num_tot += 1

# report performance
num_tot = len([item[-1] for item in all_matching_res if len(item) > 2 and item[-2] and isinstance(item[-1], int)])
rank_dict = {k: v / num_tot for k, v in dict(Counter([item[-1] for item in all_matching_res if len(item) > 2 and item[-2] and isinstance(item[-1], int)])).items()}
print(rank_dict)

# write results
json.dump(all_matching_res, open(f'pred/entities_detected.{dataset}.json', 'w'))