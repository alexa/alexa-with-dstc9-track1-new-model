from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
import nltk
# from rouge import Rouge
from rouge_score import rouge_scorer

import re

import sys
import json
import argparse
import copy

RE_ART = re.compile(r'\b(a|an|the)\b')
RE_PUNC = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

class Metric:
    def __init__(self):
        self.reset()

    def reset(self):
        self._detection_tp = 0.0
        self._detection_fp = 0.0
        self._detection_tn = 0.0
        self._detection_fn = 0.0

        self._selection_mrr5 = 0.0
        self._selection_r1 = 0.0
        self._selection_r5 = 0.0

        self._generation_bleu1 = 0.0
        self._generation_bleu2 = 0.0
        self._generation_bleu3 = 0.0
        self._generation_bleu4 = 0.0
        self._generation_meteor = 0.0
        self._generation_rouge_1 = 0.0
        self._generation_rouge_2 = 0.0
        self._generation_rouge_l = 0.0

    def _match(self, ref_knowledge, pred_knowledge):
        result = []
        for pred in pred_knowledge:
            matched = False
            for ref in ref_knowledge:
                if pred['domain'] == ref['domain'] and str(pred['entity_id']) == str(ref['entity_id']) and str(pred['doc_id']) == str(ref['doc_id']):
                    matched = True
            result.append(matched)
        return result

    def _reciprocal_rank(self, ref_knowledge, hyp_knowledge, k=5):
        relevance = self._match(ref_knowledge, hyp_knowledge)[:k]

        if True in relevance:
            idx = relevance.index(True)
            result = 1.0 / (idx + 1)
        else:
            result = 0.0

        return result

    def _recall_at_k(self, ref_knowledge, hyp_knowledge, k=5):
        relevance = self._match(ref_knowledge, hyp_knowledge)[:k]

        if True in relevance:
            result = 1.0
        else:
            result = 0.0

        return result

    def _normalize_text(self, text):
        result = text.lower()
        result = RE_PUNC.sub(' ', result)
        result = RE_ART.sub(' ', result)
        result = ' '.join(result.split())

        return result

    def _bleu(self, ref_response, hyp_response, n=4):
        ref_tokens = self._normalize_text(ref_response).split()
        hyp_tokens = self._normalize_text(hyp_response).split()

        weights = [1.0 / n] * n

        score = sentence_bleu([ref_tokens], hyp_tokens, weights)

        return score

    def _meteor(self, ref_response, hyp_response):
        score = single_meteor_score(ref_response, hyp_response, self._normalize_text)

        return score

    def _rouge(self, ref_response, hyp_response, mode='l'):
        ref_response = self._normalize_text(ref_response)
        hyp_response = self._normalize_text(hyp_response)

        if mode == 'l':
            rouge = rouge_scorer.RougeScorer(['rougeL'])
            score = rouge.score(hyp_response, ref_response)['rougeL'].fmeasure
        elif mode == 1:
            rouge = rouge_scorer.RougeScorer(['rouge1'])
            score = rouge.score(hyp_response, ref_response)['rouge1'].fmeasure
        elif mode == 2:
            rouge = rouge_scorer.RougeScorer(['rouge2'])
            score = rouge.score(hyp_response, ref_response)['rouge2'].fmeasure
        else:
            raise ValueError("unsupported mode: %s" % mode)

        return score

    def update(self, ref_obj, hyp_obj, check_whole_response=True):
        if ref_obj['target'] is True:
            if hyp_obj['target'] is True:
                self._detection_tp += 1

                self._selection_mrr5 += self._reciprocal_rank(ref_obj['knowledge'], hyp_obj['knowledge'], 5)
                self._selection_r1 += self._recall_at_k(ref_obj['knowledge'], hyp_obj['knowledge'], 1)
                self._selection_r5 += self._recall_at_k(ref_obj['knowledge'], hyp_obj['knowledge'], 5)

                # check whether remove the last sentence from response can improve
                if not check_whole_response:
                    response = nltk.sent_tokenize(ref_obj['response'])
                    ref_obj['response'] = ' '.join(response[:-1]) if len(response) > 1 else response[0]
                    response = nltk.sent_tokenize(hyp_obj['response'])
                    hyp_obj['response'] = ' '.join(response[:-1]) if len(response) > 1 else response[0]

                self._generation_bleu1 += self._bleu(ref_obj['response'], hyp_obj['response'], 1)
                self._generation_bleu2 += self._bleu(ref_obj['response'], hyp_obj['response'], 2)
                self._generation_bleu3 += self._bleu(ref_obj['response'], hyp_obj['response'], 3)
                self._generation_bleu4 += self._bleu(ref_obj['response'], hyp_obj['response'], 4)
                self._generation_meteor += self._meteor(ref_obj['response'], hyp_obj['response'])
                self._generation_rouge_l += self._rouge(ref_obj['response'], hyp_obj['response'], 'l')
                self._generation_rouge_1 += self._rouge(ref_obj['response'], hyp_obj['response'], 1)
                self._generation_rouge_2 += self._rouge(ref_obj['response'], hyp_obj['response'], 2)
            else:
                self._detection_fn += 1
        else:
            if hyp_obj['target'] is True:
                self._detection_fp += 1
            else:
                self._detection_tn += 1

    def _compute(self, score_sum):
        if self._detection_tp + self._detection_fp > 0.0:
            score_p = score_sum / (self._detection_tp + self._detection_fp)
        else:
            score_p = 0.0

        if self._detection_tp + self._detection_fn > 0.0:
            score_r = score_sum / (self._detection_tp + self._detection_fn)
        else:
            score_r = 0.0

        if score_p + score_r > 0.0:
            score_f = 2 * score_p * score_r / (score_p + score_r)
        else:
            score_f = 0.0

        return (score_p, score_r, score_f)

    def scores(self):
        detection_p, detection_r, detection_f = self._compute(self._detection_tp)

        selection_mrr5_p, selection_mrr5_r, selection_mrr5_f = self._compute(self._selection_mrr5)
        selection_r1_p, selection_r1_r, selection_r1_f = self._compute(self._selection_r1)
        selection_r5_p, selection_r5_r, selection_r5_f = self._compute(self._selection_r5)

        generation_bleu1_p, generation_bleu1_r, generation_bleu1_f = self._compute(self._generation_bleu1)
        generation_bleu2_p, generation_bleu2_r, generation_bleu2_f = self._compute(self._generation_bleu2)
        generation_bleu3_p, generation_bleu3_r, generation_bleu3_f = self._compute(self._generation_bleu3)
        generation_bleu4_p, generation_bleu4_r, generation_bleu4_f = self._compute(self._generation_bleu4)
        generation_meteor_p, generation_meteor_r, generation_meteor_f = self._compute(self._generation_meteor)
        generation_rouge_l_p, generation_rouge_l_r, generation_rouge_l_f = self._compute(self._generation_rouge_l)
        generation_rouge_1_p, generation_rouge_1_r, generation_rouge_1_f = self._compute(self._generation_rouge_1)
        generation_rouge_2_p, generation_rouge_2_r, generation_rouge_2_f = self._compute(self._generation_rouge_2)

        scores = {
            'instance number': self._detection_tp + self._detection_fp,
            'detection': {
                'prec': detection_p,
                'rec': detection_r,
                'f1': detection_f
            },
            'selection': {
                'mrr@5': selection_mrr5_f,
                'r@1': selection_r1_f,
                'r@5': selection_r5_f,
            },
            'generation': {
                'bleu-1': generation_bleu1_f,
                'bleu-2': generation_bleu2_f,
                'bleu-3': generation_bleu3_f,
                'bleu-4': generation_bleu4_f,
                'meteor': generation_meteor_f,
                'rouge_1': generation_rouge_1_f,
                'rouge_2': generation_rouge_2_f,
                'rouge_l': generation_rouge_l_f
            }
        }

        return scores

        
def main(argv):
    parser = argparse.ArgumentParser(description='Evaluate the system outputs.')

    parser.add_argument('--outfile', dest='outfile', action='store', metavar='JSON_FILE', required=True,
                        help='File containing output')
    parser.add_argument('--ksfile', dest='ksfile', action='store', metavar='JSON_FILE', required=True,
                        help='File containing knowledge selection prediction')
    parser.add_argument('--reffile', dest='reffile', action='store', metavar='JSON_FILE', required=True,
                        help='File containing reference')
    parser.add_argument('--scorefile', dest='scorefile', action='store', metavar='JSON_FILE', required=True,
                        help='File containing scores')
    parser.add_argument('--finalfile', dest='finalfile', action='store', metavar='JSON_FILE', required=True,
                        help='File to be generated in the end')

    args = parser.parse_args()

    with open(args.outfile, 'r') as f:
        output = list(map(lambda x: x.strip(), f.readlines()))
    with open(args.ksfile, 'r') as f:
        ks_preds = json.load(f)
    with open(args.reffile, 'r') as f:
        references = json.load(f)
    
    scores_all = {}

    # evaluate whole response
    check_whole_response = True
    metric = Metric()

    count = 0
    final_pred = []
    for ref, ks_pred in zip(references, ks_preds):
        if ks_pred['target']:
            res = output[count]
            ks_pred['response'] = res
            metric.update(ref, ks_pred, check_whole_response=check_whole_response)
            count += 1
            final_pred.append(copy.deepcopy(ks_pred))
        else:
            final_pred.append(copy.deepcopy(ks_pred))
            ks_pred['response'] = ''
            metric.update(ref, ks_pred, check_whole_response=check_whole_response)

    assert count == len(output)
    scores = metric.scores()
    scores_all['whole response'] = scores

    # evaluate the first part
    check_whole_response = False
    metric = Metric()

    count = 0
    for ref, ks_pred in zip(references, ks_preds):
        if ks_pred['target']:
            res = output[count]
            ks_pred['response'] = res
            metric.update(ref, ks_pred, check_whole_response=check_whole_response)
            count += 1
        else:
            ks_pred['response'] = ''
            metric.update(ref, ks_pred, check_whole_response=check_whole_response)

    scores = metric.scores()
    scores_all['first part'] = scores

    print(scores_all)
    with open(args.scorefile, 'w') as out:
        json.dump(scores_all, out, indent=2)
    with open(args.finalfile, 'w') as out:
        json.dump(final_pred, out, indent=2)


if __name__ =="__main__":
    main(sys.argv)        
