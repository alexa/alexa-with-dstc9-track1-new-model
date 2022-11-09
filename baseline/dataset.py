import os
import json
import random
import logging
import sys
from collections import Counter, defaultdict
import numpy as np

from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack

import torch

from tqdm import tqdm

from .utils.data import (
    pad_ids, truncate_sequences, make_input_masks
)

from scripts.dataset_walker import DatasetWalker, processors
from scripts.knowledge_reader import KnowledgeReader

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>"],
}
SPECIAL_TOKENS_VALUES = ["<bos>", "<eos>", "<pad>", "<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>"]


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None,
                 entities_file=None, domain_list=None):
        self.args = args
        self.dataroot = args.dataroot
        self.tokenizer = tokenizer
        self.split_type = split_type

        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.SPECIAL_TOKENS_VALUES = SPECIAL_TOKENS_VALUES
        if not hasattr(args, "model_for_ks") or 'gpt2' in args.model_for_ks:
            self.bos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["bos_token"])
            self.eos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["eos_token"])
            self.pad = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["pad_token"])
        else:
            self.bos = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
            self.eos = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
            self.pad = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        self.speaker1, self.speaker2, self.knowledge_sep, self.knowledge_tag = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["additional_special_tokens"]
        )
        self.knowledge_sep_token = self.SPECIAL_TOKENS["additional_special_tokens"][2]

        self.dataset_walker = DatasetWalker(split_type, labels=labels, dataroot=self.dataroot, labels_file=labels_file)
        self.dialogs = self._prepare_conversations(domain_list=domain_list)

        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')

        self.knowledge_reader = KnowledgeReader(self.dataroot, args.knowledge_file)
        self.snippets, self.snippets_for_ks, self.snippets_entities, \
        self.snippets_categories, self.snippets_tfidf_vecs \
            = self._prepare_knowledge(domain_list=domain_list)

        # read entities file if it exists
        if entities_file is not None:
            self.detected_entities = json.load(open(entities_file))
        else:
            self.detected_entities = None

        if hasattr(self.args, "negative_sample_method") and \
                self.args.negative_sample_method == 'mix' and \
                hasattr(self.args, "train_detected_entities_file") \
                and self.args.train_detected_entities_file:
            self.train_detected_entites = json.load(open(self.args.train_detected_entities_file))

        self._create_examples()
        # self.examples = self.examples[:20]

    def _prepare_conversations(self, domain_list=None):
        logger.info("Tokenize and encode the dialog data")
        tokenized_dialogs = []
        for i, (log, label) in enumerate(tqdm(self.dataset_walker, disable=self.args.local_rank not in [-1, 0])): # only show progress bar in one process
            if domain_list is not None and label['target']:
                if label['knowledge'][0]['domain'] not in domain_list:
                    continue
            dialog = {}
            dialog["id"] = i
            dialog["log"] = log
            if label is not None:
                if "response" in label:
                    label["response_tokenized"] = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(label["response"])
                    )
            dialog["label"] = label
            tokenized_dialogs.append(dialog)
        return tokenized_dialogs
    
    def _prepare_knowledge(self, domain_list=None):
        # knowledge = self.knowledge_reader.knowledge
        if domain_list is not None:
            self.knowledge_docs = list(chain(*[self.knowledge_reader.get_doc_list(domain=domain)
                                               for domain in domain_list]))
        else:
            self.knowledge_docs = self.knowledge_reader.get_doc_list()

        tokenized_snippets = dict()
        tokenized_snippets_for_ks = dict()
        tokenized_entities = dict()
        categories = dict()
        key_ls, knowledge_str_ls = [], []
        self.key_mapping_entity_id_to_doc_id = defaultdict(list)

        for snippet in self.knowledge_docs:
            entity_key = "{}__{}".format(snippet["domain"], str(snippet["entity_id"]))
            key = "{}__{}__{}".format(snippet["domain"], str(snippet["entity_id"]), snippet["doc_id"])
            self.key_mapping_entity_id_to_doc_id[entity_key].append(key)

            # knowledge = self._knowledge_to_string(snippet["doc"], name=snippet["entity_name"] or "")
            body = snippet["doc"]["body"]
            title = snippet["doc"]["title"]
            domain_name = snippet["domain"]
            entity_name = snippet["entity_name"] if snippet["entity_name"] else ''
            tokenized_knowledge = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(body))
            tokenized_snippets[key] = tokenized_knowledge[:self.args.knowledge_max_tokens]
            categories[key] = snippet['doc']['category']

            entity_key = "{}__{}__{}".format(snippet["domain"], str(snippet["entity_id"]), "*")
            tokenized_entities[entity_key] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(entity_name))

            if hasattr(self.args, "knowledge_selection_method"):
                if self.args.knowledge_selection_method.startswith('ans'):
                    knowledge = body
                elif self.args.knowledge_selection_method.startswith('ques'):
                    knowledge = title
                elif self.args.knowledge_selection_method.startswith('qa'):
                    knowledge = title + ' ' + body
                else:
                    raise NotImplementedError
                if self.args.knowledge_selection_method.endswith('entity'):
                    knowledge = domain_name + ' : ' + entity_name + ' : ' + knowledge
                tokenized_knowledge = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(knowledge))
                tokenized_snippets_for_ks[key] = tokenized_knowledge[:self.args.knowledge_max_tokens]
                knowledge_str_ls.append(knowledge) # for TF-IDF
                key_ls.append(key)
            else:
                tokenized_snippets_for_ks = tokenized_snippets.copy()

        # build the TF-IDF vectorizer
        snippets_tfidf_vecs = self.tfidf_vectorizer.fit_transform(knowledge_str_ls)
        snippets_tfidf_vecs = {key: vec for key, vec in zip(key_ls, snippets_tfidf_vecs)}

        return tokenized_snippets, tokenized_snippets_for_ks, tokenized_entities, categories, snippets_tfidf_vecs

    def _knowledge_to_string(self, doc, name=""):
        return doc["body"]

    def _create_examples(self):
        logger.info("Creating examples")
        self.examples = []
        for dialog in tqdm(self.dialogs, disable=self.args.local_rank not in [-1, 0]):
            dialog_id = dialog["id"]
            label = dialog["label"]
            dialog = dialog["log"]
            if label is None:
                # This will only happen when running knowledge-seeking turn detection on test data
                # So we create dummy target here
                label = {"target": False}

            target = label["target"]

            if not target and self.args.task not in ['detection', 'binary-classification']:
                # we only care about non-knowledge-seeking turns in turn detection task
                continue
            
            history = [
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(turn["text"]))
                for turn in dialog
            ]
            gt_resp = label.get("response", "")
            tokenized_gt_resp = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(gt_resp))

            # apply history threshold at an utterance-level (a large value can be used to nullify its effect)
            truncated_history = history[-self.args.history_max_utterances:]

            # perform token-level truncation of history from the left
            truncated_history = truncate_sequences(truncated_history, self.args.history_max_tokens)

            if target and self.args.task not in ['domain-detection', 'binary-classification']:
                if "knowledge" not in label:
                    # when the labels.json is from knowledge-seeking turn detection,
                    # there will be no ground truth knowledge
                    # so we just use a dummy snippet here
                    # if not self.args.eval_all_snippets:
                    #     raise ValueError("eval_all_snippets is required to be true when taking output from knowledge-seeking turn detection")
                    label["knowledge"] = [self.knowledge_docs[0]]

                knowledge = label["knowledge"][0]
                knowledge_key = "{}__{}__{}".format(knowledge["domain"], knowledge["entity_id"], knowledge["doc_id"])

                if self.detected_entities is None:
                    # find snippets with same entity as candidates
                    prefixes = ["{}__{}".format(knowledge["domain"], knowledge["entity_id"])]
                else:
                    prefixes = ["{}__{}".format(domain, entity_id)
                                for (domain, entity_id) in self.detected_entities[dialog_id]]
                if len(self.key_mapping_entity_id_to_doc_id) == 1:
                    knowledge_candidates = self.snippets.keys()
                else:
                    knowledge_candidates = []
                    for prefix in prefixes:
                        knowledge_candidates.extend(self.key_mapping_entity_id_to_doc_id[prefix])
                # knowledge_candidates = [
                #     cand
                #     for prefix in prefixes for cand in self.snippets.keys()
                #     if "__".join(cand.split("__")[:-1]) == prefix
                # ]
                if self.split_type == "train" and self.args.negative_sample_method == "oracle":
                    # if there's not enough candidates during training, we just skip this example
                    if len(knowledge_candidates) < self.args.n_candidates:
                        continue
                try:
                    used_knowledge = self.snippets[knowledge_key]
                    used_knowledge_for_ks = self.snippets_for_ks[knowledge_key]
                except:
                    print(knowledge_key)
                    used_knowledge = []
                    used_knowledge_for_ks = []
                # used_knowledge = used_knowledge[:self.args.knowledge_max_tokens]
            else:
                knowledge_candidates = None
                used_knowledge = []
                used_knowledge_for_ks = []

            self.examples.append({
                "history": truncated_history,
                "knowledge": used_knowledge,
                "knowledge_for_ks": used_knowledge_for_ks,
                "candidates": knowledge_candidates,
                "response": tokenized_gt_resp,
                "response_text": gt_resp,
                "label": label,
                "knowledge_seeking": target,
                "dialog_id": dialog_id,
                "last_user_utt_tfidf_vec": self.tfidf_vectorizer.transform([dialog[-1]['text']])
            })

    def build_input_from_segments(self, knowledge, history, response, with_eos=True):
        """ Build a sequence of input from 3 segments: knowledge, history and last reply """
        instance = {}

        sequence = [[self.bos] + knowledge] + history + [response + ([self.eos] if with_eos else [])]
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        sequence = [sequence[0]] + sequence_with_speaker
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence) for _ in s]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]

        return instance, sequence
                
    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.examples)


class ResponseGenerationDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None, entities_file=None):
        super(ResponseGenerationDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = self.examples[index]
        instance, _ = self.build_input_from_segments(
            example["knowledge"],
            example["history"],
            example["response"]
        )
        return instance

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        token_type_ids = [ins["token_type_ids"] for ins in batch]
        lm_labels = [ins["lm_labels"] for ins in batch]

        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        token_type_ids = torch.tensor(pad_ids(token_type_ids, self.pad))
        lm_labels = torch.tensor(pad_ids(lm_labels, -100))

        return input_ids, token_type_ids, lm_labels


class ResponseGenerationEvalDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None, entities_file=None):
        super(ResponseGenerationEvalDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = self.examples[index]
        return example

    def collate_fn(self, batch):
        return batch


class KnowledgeSelectionDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None, entities_file=None):
        super(KnowledgeSelectionDataset, self).__init__(args, tokenizer, split_type, labels, labels_file, entities_file)
        if self.args.negative_sample_method not in ["all", "mix", "oracle"]:
            raise ValueError("negative_sample_method must be all, mix, or oracle, got %s" % self.args.negative_sample_method)

    def _knowledge_to_string(self, doc, name=""):
        join_str = " %s " % self.knowledge_sep_token
        return join_str.join([name, doc["title"], doc["body"]])

    def __getitem__(self, index):
        example = self.examples[index]

        this_inst = {
            "dialog_id": example["dialog_id"],
            "input_ids": [],
            "token_type_ids": [],
            "mc_token_ids": []
        }

        if self.split_type != "train":
            # if eval_all_snippets is set, we use all snippets as candidates
            if self.args.eval_all_snippets:
                candidates_keys = list(self.snippets_for_ks.keys())
            else:
                candidates_keys = example["candidates"]
        else:
            if self.args.negative_sample_method == "all":
                candidates_keys = list(self.snippets_for_ks.keys())

                # now we need decide whether random sample the negative entities or rank them by TF-IDF similarity
                use_tfidf = random.choices(range(2), [1 - self.args.use_tfidf_rate, self.args.use_tfidf_rate],
                                           k=1)[0]
                # use_tfidf = 0
                if use_tfidf:
                    # this part can return the ranking of TF-IDF similarity and we use the top-random_state_count
                    example_vec = example['last_user_utt_tfidf_vec']
                    candidates_vecs = [self.snippets_tfidf_vecs[entity_key] for entity_key in candidates_keys]
                    cos_sim_scores = cosine_similarity(example_vec, vstack(candidates_vecs))[0]
                    candidates_keys = [candidates_keys[idx] for idx in
                                       np.argsort(cos_sim_scores)[-self.args.n_candidates:]]

            elif self.args.negative_sample_method == "mix":
                negative_samples = []
                label = "{}__{}__{}".format(example['label']['knowledge'][0]['domain'],
                                            example['label']['knowledge'][0]['entity_id'],
                                            example['label']['knowledge'][0]['doc_id'])
                label_entity = "{}__{}".format(example['label']['knowledge'][0]['domain'],
                                            example['label']['knowledge'][0]['entity_id'])

                # these entities exist in the knowledge base and appear in the same dialogue as the gold entity
                candidates_entity = ["{}__{}".format(entity[0], entity[1])
                                     for entity in self.train_detected_entites[int(example["dialog_id"])][0]]
                # argument of negative_mix_percent controls the probability of those negative sampling strategies used
                if candidates_entity:
                    random_states = random.choices(range(len(self.args.negative_mix_percent)),
                                                   self.args.negative_mix_percent, k=self.args.n_candidates - 1)
                else:
                    random_states = random.choices(range(len(self.args.negative_mix_percent[:-1])),
                                                   self.args.negative_mix_percent[:-1], k=self.args.n_candidates - 1)

                random_states = dict(Counter(random_states))
                for random_state, random_state_count in random_states.items():
                    if random_state == 0:
                        # strategy 1: select negative samples from all knowledge snippets
                        entity_candidates = list(self.snippets_for_ks.keys())
                    elif random_state == 1:
                        # strategy 2: select negative samples from knowledge snippets of entities in the same domain
                        # as the gold entity
                        entity_candidates = example['candidates']
                    elif random_state == 2:
                        # strategy 3: select negative samples from knowledge snippets of the gold entity
                        prefix = example['label']['knowledge'][0]['domain']
                        entity_candidates = [
                            cand
                            for cand in self.snippets_for_ks.keys()
                            if cand.split("__")[0] == prefix
                        ]
                    elif random_state == 3:
                        # strategy 4: select negative samples from knowledge snippets of entities that
                        # appear in the same dialogue as the gold one
                        try:
                            negative_entity = self._random_select_negatives(label_entity, candidates_entity, 1)[0]
                        except:
                            negative_entity = random.sample(candidates_entity, 1)[0]

                        entity_candidates = [
                            cand
                            for cand in self.snippets_for_ks.keys()
                            if "__".join(cand.split("__")[:-1]) == negative_entity
                        ]
                    else:
                        raise NotImplementedError

                    # now we need decide whether random sample the negative entities or rank them by TF-IDF similarity
                    use_tfidf = random.choices(range(2), [1-self.args.use_tfidf_rate, self.args.use_tfidf_rate],
                                               k=1)[0]
                    # use_tfidf = 0
                    if use_tfidf:
                        # this part can return the ranking of TF-IDF similarity and we use the top-random_state_count
                        example_vec = example['last_user_utt_tfidf_vec']
                        candidates_vecs = [self.snippets_tfidf_vecs[entity_key] for entity_key in entity_candidates]
                        cos_sim_scores = cosine_similarity(example_vec, vstack(candidates_vecs))[0]
                        negative_samples += [entity_candidates[idx] for idx in np.argsort(cos_sim_scores)[-random_state_count:]]
                    else:
                        negative_samples += random.sample(entity_candidates, random_state_count)

                candidates_keys = [label] + negative_samples
                random.shuffle(candidates_keys)
                assert len(candidates_keys) == self.args.n_candidates
            elif self.args.negative_sample_method == "oracle":
                candidates_keys = example["candidates"]
            else: # although we have already checked for this, still adding this here to be sure
                raise ValueError("negative_sample_method must be all, mix, or oracle, got %s" % self.args.negative_sample_method)
        
        this_inst["candidate_keys"] = candidates_keys
        candidates = [self.snippets_for_ks[cand_key] for cand_key in candidates_keys]

        if self.split_type == "train" and len(candidates) > self.args.n_candidates and \
                self.args.negative_sample_method != "mix":
            candidates = self._shrink_label_cands(example["knowledge_for_ks"], candidates)

        try:
            label_idx = candidates.index(example["knowledge_for_ks"])
        except:
            label_idx = 0
            
        this_inst["label_idx"] = label_idx

        for cand in candidates:
            instance, _ = self.build_input_from_segments(
                cand,
                example["history"][-self.args.history_sent_num_for_ks:]
                if hasattr(self.args, 'history_sent_num_for_ks') and self.args.history_sent_num_for_ks > 0
                else example["history"],
                for_gpt='gpt2' in self.args.model_for_ks
            )
            this_inst["input_ids"].append(instance["input_ids"])
            this_inst["token_type_ids"].append(instance["token_type_ids"])
            this_inst["mc_token_ids"].append(instance["mc_token_ids"])

        return this_inst

    def build_input_from_segments(self, knowledge, history, for_gpt=True):
        """ Build a sequence of input from 2 segments: knowledge and history"""
        instance = {}

        sequence = [[self.bos]] + history
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]

        if for_gpt:
            sequence = [sequence[0]] + sequence_with_speaker + [[self.knowledge_tag] + knowledge + [self.eos]]
            instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence[:-1]) for _ in s] + [self.knowledge_tag for _ in sequence[-1]]
        else:
            sequence = [sequence[0]] + sequence_with_speaker + [[self.eos] + knowledge + [self.eos]]
            instance["token_type_ids"] = [0 for s in sequence[:-1] for _ in s] + [1 for _ in sequence[-1]] # for bert like models

        instance["input_ids"] = list(chain(*sequence))
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1

        assert len(instance["input_ids"]) == len(instance["token_type_ids"])

        # if len(instance["input_ids"]) == 0:
        #     print(knowledge, history)

        return instance, sequence
    
    def _shrink_label_cands(self, label, candidates, shrink_num=None):
        if shrink_num is None:
            shrink_num = self.args.n_candidates - 1
        shrunk_label_cands = candidates.copy()
        shrunk_label_cands.remove(label)
        shrunk_label_cands = random.sample(shrunk_label_cands, k=shrink_num)
        shrunk_label_cands.append(label)
        random.shuffle(shrunk_label_cands)

        return shrunk_label_cands

    def _random_select_negatives(self, label, candidates, shrink_num=None):
        if shrink_num is None:
            shrink_num = self.args.n_candidates - 1
        shrunk_label_cands = candidates.copy()
        shrunk_label_cands.remove(label)
        shrunk_label_cands = random.sample(shrunk_label_cands, k=shrink_num)
        random.shuffle(shrunk_label_cands)

        return shrunk_label_cands

    def collate_fn(self, batch):
        input_ids = [ids for ins in batch for ids in ins["input_ids"]]
        token_type_ids = [ids for ins in batch for ids in ins["token_type_ids"]]
        mc_token_ids = [id for ins in batch for id in ins["mc_token_ids"]]
        label_idx = [ins["label_idx"] for ins in batch]

        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch],
            "candidate_keys": [ins["candidate_keys"] for ins in batch]
        }
        # for ins in batch:
        #     print(ins["dialog_id"])
        #     print(ins["candidate_keys"])
        #     print('\n\n')

        batch_size = len(batch)
        n_candidates = len(batch[0]["input_ids"])

        input_masks = torch.tensor(
            make_input_masks(input_ids)
        ).view(batch_size, n_candidates, -1)

        input_ids = torch.tensor(
            pad_ids(input_ids, self.pad)
        ).view(batch_size, n_candidates, -1)
        
        token_type_ids = torch.tensor(
            pad_ids(token_type_ids, 0)
        ).view(batch_size, n_candidates, -1)

        assert input_ids.shape == token_type_ids.shape
        assert input_ids.shape == input_masks.shape

        lm_labels = torch.full_like(input_ids, -100)
        mc_token_ids = torch.tensor(mc_token_ids).view(batch_size, n_candidates)
        label_idx = torch.tensor(label_idx)

        return input_ids, token_type_ids, input_masks, mc_token_ids, lm_labels, label_idx, data_info


class EntitySelectionDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None, entities_file=None, domain_list=None):
        super(EntitySelectionDataset, self).__init__(args, tokenizer, split_type, labels, labels_file, entities_file, domain_list)
        if self.args.negative_sample_method not in ["all", "mix", "oracle"]:
            raise ValueError(
                "negative_sample_method must be all, mix, or oracle, got %s" % self.args.negative_sample_method)

    def _knowledge_to_string(self, doc, name=""):
        join_str = " %s " % self.knowledge_sep_token
        return join_str.join([name, doc["title"], doc["body"]])

    def __getitem__(self, index):
        example = self.examples[index]

        this_inst = {
            "dialog_id": example["dialog_id"],
            "input_ids": [],
            "token_type_ids": [],
            "mc_token_ids": []
        }

        candidate_keys = list(self.snippets_entities.keys())
        this_inst["candidate_keys"] = candidate_keys
        label = "{}__{}__*".format(example['label']['knowledge'][0]['domain'],
                                   example['label']['knowledge'][0]['entity_id'])

        if self.split_type == "train":
            candidate_keys = self._shrink_label_cands(label, candidate_keys)

        label_idx = candidate_keys.index(label)
        this_inst["label_idx"] = label_idx

        candidates = [self.snippets_entities[cand_key] for cand_key in candidate_keys]

        for cand in candidates:
            instance, _ = self.build_input_from_segments(
                cand,
                example["history"][-self.args.history_sent_num_for_ks:]
                if hasattr(self.args, 'history_sent_num_for_ks') and self.args.history_sent_num_for_ks > 0
                else example["history"],
                for_gpt='gpt2' in self.args.model_for_ks
            )
            this_inst["input_ids"].append(instance["input_ids"])
            this_inst["token_type_ids"].append(instance["token_type_ids"])
            this_inst["mc_token_ids"].append(instance["mc_token_ids"])

        return this_inst

    def build_input_from_segments(self, knowledge, history, for_gpt=True):
        """ Build a sequence of input from 2 segments: knowledge and history"""
        instance = {}

        sequence = [[self.bos]] + history
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]

        if for_gpt:
            sequence = [sequence[0]] + sequence_with_speaker + [[self.knowledge_tag] + knowledge + [self.eos]]
            instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence[:-1])
                                          for _ in s] + [self.knowledge_tag for _ in sequence[-1]]
        else:
            sequence = [sequence[0]] + sequence_with_speaker + [[self.eos] + knowledge + [self.eos]]
            instance["token_type_ids"] = [0 for s in sequence[:-1] for _ in s] + [1 for _ in
                                                                                  sequence[-1]]  # for bert like models

        instance["input_ids"] = list(chain(*sequence))
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1

        assert len(instance["input_ids"]) == len(instance["token_type_ids"])

        return instance, sequence

    def _shrink_label_cands(self, label, candidates):
        shrunk_label_cands = candidates.copy()
        shrunk_label_cands.remove(label)
        shrunk_label_cands = random.sample(shrunk_label_cands, k=self.args.n_candidates - 1)
        shrunk_label_cands.append(label)
        random.shuffle(shrunk_label_cands)

        return shrunk_label_cands

    def collate_fn(self, batch):
        input_ids = [ids for ins in batch for ids in ins["input_ids"]]
        token_type_ids = [ids for ins in batch for ids in ins["token_type_ids"]]
        mc_token_ids = [id for ins in batch for id in ins["mc_token_ids"]]
        label_idx = [ins["label_idx"] for ins in batch]

        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch],
            "candidate_keys": [ins["candidate_keys"] for ins in batch]
        }
        # for ins in batch:
        #     print(ins["dialog_id"])
        #     print(ins["candidate_keys"])
        #     print('\n\n')

        batch_size = len(batch)
        n_candidates = len(batch[0]["input_ids"])

        input_masks = torch.tensor(
            make_input_masks(input_ids)
        ).view(batch_size, n_candidates, -1)

        input_ids = torch.tensor(
            pad_ids(input_ids, self.pad)
        ).view(batch_size, n_candidates, -1)

        token_type_ids = torch.tensor(
            pad_ids(token_type_ids, 0)
        ).view(batch_size, n_candidates, -1)

        assert input_ids.shape == token_type_ids.shape
        assert input_ids.shape == input_masks.shape

        lm_labels = torch.full_like(input_ids, -100)
        mc_token_ids = torch.tensor(mc_token_ids).view(batch_size, n_candidates)
        label_idx = torch.tensor(label_idx)

        return input_ids, token_type_ids, input_masks, mc_token_ids, lm_labels, label_idx, data_info


class KnowledgeTurnDetectionDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None, entities_file=None):
        super(KnowledgeTurnDetectionDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def build_input_from_segments(self, history, for_gpt=True):
        """ Build a sequence of input from history """
        instance = {}

        sequence = [[self.bos]] + history[:-1] + [[self.knowledge_tag] + history[-1] + [self.eos]]
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        sequence = [sequence[0]] + sequence_with_speaker

        if for_gpt:
            instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence) for _ in s]
        else:
            instance["token_type_ids"] = [0 for s in sequence[:-1] for _ in s] + [1 for _ in sequence[-1]] # for bert like models

        instance["input_ids"] = list(chain(*sequence))
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1

        return instance, sequence

    def __getitem__(self, index):
        example = self.examples[index]
        instance, _ = self.build_input_from_segments(
            example["history"][-self.args.history_sent_num_for_ks:]
            if hasattr(self.args, 'history_sent_num_for_ks') and self.args.history_sent_num_for_ks > 0
            else example["history"],
            for_gpt='gpt2' in self.args.model_for_ks
        )
        instance["label"] = example["knowledge_seeking"]
        instance["dialog_id"] = example["dialog_id"]
        return instance

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        token_type_ids = [ins["token_type_ids"] for ins in batch]
        mc_token_ids = [ins["mc_token_ids"] for ins in batch]
        labels = [ins["label"] for ins in batch]

        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch]
        }

        input_masks = torch.tensor(
            make_input_masks(input_ids)
        )
        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        token_type_ids = torch.tensor(pad_ids(token_type_ids, self.pad))
        mc_token_ids = torch.tensor(mc_token_ids)
        lm_labels = torch.full_like(input_ids, -100)
        labels = torch.tensor(labels).long()

        return input_ids, token_type_ids, input_masks, mc_token_ids, lm_labels, labels, data_info


class DomainDetectionDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None, entities_file=None):
        super(DomainDetectionDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)
        self.all_domains = ['train', 'taxi', 'others']

    def build_input_from_segments(self, history, for_gpt=True):
        """ Build a sequence of input from history """
        instance = {}

        sequence = [[self.bos]] + history[:-1] + [[self.eos] + history[-1] + [self.eos]]
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        sequence = [sequence[0]] + sequence_with_speaker

        if for_gpt:
            instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence) for _ in s]
        else:
            instance["token_type_ids"] = [0 for s in sequence[:-1] for _ in s] + [1 for _ in sequence[-1]] # for bert like models

        instance["input_ids"] = list(chain(*sequence))
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1

        return instance, sequence

    def __getitem__(self, index):
        example = self.examples[index]

        instance, _ = self.build_input_from_segments(
            example["history"][-self.args.history_sent_num_for_ks:]
            if hasattr(self.args, 'history_sent_num_for_ks') and self.args.history_sent_num_for_ks > 0
            else example["history"],
            for_gpt='gpt2' in self.args.model_for_ks
        )
        try:
            instance["label"] = self.all_domains.index(example["label"]["knowledge"][0]["domain"])
        except:
            # this will happen for test set when label file is provided
            instance["label"] = 2
        instance["dialog_id"] = example["dialog_id"]
        return instance

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        token_type_ids = [ins["token_type_ids"] for ins in batch]
        mc_token_ids = [ins["mc_token_ids"] for ins in batch]
        labels = [ins["label"] for ins in batch]

        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch]
        }

        input_masks = torch.tensor(
            make_input_masks(input_ids)
        )
        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        token_type_ids = torch.tensor(pad_ids(token_type_ids, self.pad))
        mc_token_ids = torch.tensor(mc_token_ids)
        lm_labels = torch.full_like(input_ids, -100)
        labels = torch.tensor(labels).long()

        return input_ids, token_type_ids, input_masks, mc_token_ids, lm_labels, labels, data_info


class SequenceClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None, entities_file=None):
        assert args.dataset_dir, "For sequence classification, the dataset should be specified"
        self.tokenizer = tokenizer

        if not hasattr(args, "model_for_ks") or 'gpt2' in args.model_for_ks:
            self.bos = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["bos_token"])
            self.eos = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["eos_token"])
            self.pad = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["pad_token"])
        else:
            self.bos = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
            self.eos = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
            self.pad = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

        dataset = args.dataset_dir.split('/')[-1].lower()
        processor = processors[dataset]()
        if split_type == 'train':
            self.examples = processor.get_train_examples(args.dataset_dir)
        else:
            self.examples = processor.get_dev_examples(args.dataset_dir)

    def build_input_from_segments(self, text_a, text_b):
        """ Build a sequence of input from history """
        instance = {}

        text_a = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text_a.lower()))
        text_b = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text_b.lower()))

        sequence = [[self.bos] + text_a] + [[self.eos] + text_b + [self.eos]]

        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [0 for _ in sequence[0]] + [1 for _ in sequence[-1]]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1

        return instance

    def __getitem__(self, index):
        example = self.examples[index]
        instance = self.build_input_from_segments(example.text_a, example.text_b)
        instance["label"] = example.label
        return instance

    def __len__(self):
        return len(self.examples)

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        token_type_ids = [ins["token_type_ids"] for ins in batch]
        mc_token_ids = [ins["mc_token_ids"] for ins in batch]
        labels = [ins["label"] for ins in batch]

        data_info = {}

        input_masks = torch.tensor(
            make_input_masks(input_ids)
        )
        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        token_type_ids = torch.tensor(pad_ids(token_type_ids, self.pad))
        mc_token_ids = torch.tensor(mc_token_ids)
        lm_labels = torch.full_like(input_ids, -100)
        labels = torch.tensor(labels).long()

        return input_ids, token_type_ids, input_masks, mc_token_ids, lm_labels, labels, data_info