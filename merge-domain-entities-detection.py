import sys
import json

dataroot = sys.argv[1]
dataset = sys.argv[2]
ktd_pred_file = sys.argv[3]
domain_pred_file = sys.argv[4]
entities_pred_file = sys.argv[5]

domain_preds = json.load(open(domain_pred_file))
all_domains = ['train', 'taxi', 'others']
domain_preds = [all_domains[pred] for pred in domain_preds]

# here read the detected entities
detected_entities = json.load(open(entities_pred_file))
assert len(detected_entities) == len(domain_preds)

knowledges = json.load(open(f'{dataroot}/knowledge.json'))

# read knowledge seeking turn detection
ktd_preds = json.load(open(ktd_pred_file))

# get all entity names
all_entity_names = []
for domain, domain_dict in knowledges.items():
    if domain in ['train', 'taxi']:
        continue
    for doc_id, docs in domain_dict.items():
        all_entity_names.append((domain, doc_id))

# here we combine the above two preds into the final file
count = 0
out = []
for ktd_pred in ktd_preds:
    if ktd_pred['target']:
        domain_pred = domain_preds[count]
        detected_entity = detected_entities[count]
        if domain_pred in ['train', 'taxi']:
            out.append([(domain_pred, '*')])
        else:
            if detected_entity[0]:
                out.append([item[:2] for item in detected_entity[0][-3:]])
            else:
                out.append(all_entity_names)
                # print(domain_pred, detected_entity)
        count += 1
    else:
        out.append([])
assert count == len(domain_preds)
json.dump(out, open(f'pred/entities_detected.{dataset}.final.json', 'w'))