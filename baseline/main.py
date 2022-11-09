import argparse
import glob
import logging
import os
import random
import shutil
import json

from typing import Dict, List, Tuple
from argparse import Namespace

import numpy as np
import sklearn
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    GPT2DoubleHeadsModel,
    GPT2LMHeadModel,
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    RobertaForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from .dataset import (
    ResponseGenerationDataset,
    KnowledgeSelectionDataset,
    EntitySelectionDataset,
    KnowledgeTurnDetectionDataset,
    SequenceClassificationDataset,
    DomainDetectionDataset,
    SPECIAL_TOKENS
)
from .models import GPT2ClsDoubleHeadsModel, RobertaForMultipleChoice, RobertaForBinarySequenceClassification
from .utils.argument import (
    set_default_params,
    set_default_dataset_params,
    update_additional_params,
    verify_args
)
from .utils.model import (
    run_batch_detection,
    run_batch_generation,
    run_batch_selection_train,
    run_batch_selection_eval,
    run_batch_classification
)
from .utils.data import write_selection_preds, write_detection_preds
from scripts.dataset_walker import processors


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

MODEL_HUB = {
    "detection":
        {
            "gpt2": GPT2DoubleHeadsModel,
            'bert': BertForSequenceClassification,
            'distilbert': DistilBertForSequenceClassification,
            "roberta": RobertaForSequenceClassification,
        },
    "selection":
        {
            "gpt2": GPT2DoubleHeadsModel,
            "roberta": RobertaForMultipleChoice,
            "distilroberta": RobertaForMultipleChoice,
        },
    "entity-selection":
        {
            "gpt2": GPT2DoubleHeadsModel,
            "roberta": RobertaForMultipleChoice,
        },
    "binary-classification":
        {
            "gpt2": GPT2ClsDoubleHeadsModel,
            "roberta": RobertaForBinarySequenceClassification,
        },
    "classification":
        {
            "gpt2": GPT2ClsDoubleHeadsModel,
            "roberta": RobertaForSequenceClassification,
        },
    "domain-detection":
        {
            "gpt2": GPT2ClsDoubleHeadsModel,
            "roberta": RobertaForSequenceClassification,
        },
    "generation":
        {
            "gpt2": GPT2LMHeadModel
        }
}

def get_classes(task, model_for_ks):
    if task.lower() == "generation":
        return ResponseGenerationDataset, MODEL_HUB[task][model_for_ks.split('-')[0]], \
               run_batch_generation, run_batch_generation
    elif task.lower() == "selection":
        return KnowledgeSelectionDataset, MODEL_HUB[task][model_for_ks.split('-')[0]],\
               run_batch_selection_train, run_batch_selection_eval
    elif task.lower() == "entity-selection":
        return EntitySelectionDataset, MODEL_HUB[task][model_for_ks.split('-')[0]],\
               run_batch_selection_train, run_batch_selection_eval
    elif task.lower() == "detection":
        return KnowledgeTurnDetectionDataset, MODEL_HUB[task][model_for_ks.split('-')[0]],\
               run_batch_detection, run_batch_detection
    elif task.lower() == 'classification':
        return SequenceClassificationDataset, MODEL_HUB[task][model_for_ks.split('-')[0]],\
                run_batch_classification, run_batch_classification
    elif task.lower() == 'domain-detection':
        return DomainDetectionDataset, MODEL_HUB[task][model_for_ks.split('-')[0]],\
                run_batch_classification, run_batch_classification
    elif task.lower() == 'binary-classification':
        return KnowledgeTurnDetectionDataset, MODEL_HUB[task][model_for_ks.split('-')[0]],\
                run_batch_detection, run_batch_detection
    else:
        raise ValueError("args.task not in ['generation', 'selection', 'detection'], got %s" % task)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, run_batch_fn_train, run_batch_fn_eval) -> Tuple[int, float]:
    if args.local_rank in [-1, 0]:
        log_dir = os.path.join("runs", args.exp_name) if args.exp_name else None
        tb_writer = SummaryWriter(log_dir)
        args.output_dir = log_dir

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=train_dataset.collate_fn,
    )

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    global_step = 0
    model.zero_grad()
    train_iterator = trange(
        0, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # for reproducibility

    for _ in train_iterator:
        local_steps = 0
        tr_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            if local_steps > 0:
                epoch_iterator.set_description("Loss: {}".format(tr_loss / local_steps))
            model.train()
            loss, _, _, _ = run_batch_fn_train(args, model, batch)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                local_steps += 1
                epoch_iterator.set_postfix(Loss=tr_loss/local_steps)

        results = evaluate(args, eval_dataset, model, tokenizer, run_batch_fn_eval, desc=str(global_step))

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / local_steps


def evaluate(args, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, run_batch_fn, desc="") -> Dict:
    if args.local_rank in [-1, 0]:
        eval_output_dir = args.output_dir
        os.makedirs(eval_output_dir, exist_ok=True)

    # eval_batch_size for selection must be 1 to handle variable number of candidates
    if args.task == "selection":
        args.eval_batch_size = 1
    else:
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=eval_dataset.collate_fn
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and (args.task != "selection" or eval_dataset.args.eval_all_snippets):
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    data_infos = []
    all_preds = []
    all_labels = []
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        with torch.no_grad():
            loss, lm_logits, mc_logits, mc_labels = run_batch_fn(args, model, batch)
            if args.task == "detection" or "binary-classification":
                mc_logits = mc_logits.sigmoid()
            if args.task in ["selection", "detection", "entity-selection"]:
                data_infos.append(batch[-1])
            all_preds.append(mc_logits.detach().cpu().numpy())
            all_labels.append(mc_labels.detach().cpu().numpy())
            eval_loss += loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps

    if args.task.lower() == "generation":
        perplexity = torch.exp(torch.tensor(eval_loss))
        result = {"perplexity": perplexity, "loss": eval_loss}
    elif args.task.lower() in ["selection", "entity-selection"]:
        all_labels = np.array(all_labels).reshape(-1)
        all_pred_ids = np.array([np.argmax(logits) for logits in all_preds])
        accuracy = np.sum(all_pred_ids == all_labels) / len(all_labels)
        logger.info("Avg. # of candidates: %f", sum([len(arr[0]) for arr in all_preds]) / len(all_preds))
        result = {"loss": eval_loss, "accuracy": accuracy}
        if args.output_file:
            sorted_pred_ids = [np.argsort(logits.squeeze())[::-1] for logits in all_preds]
            sorted_pred_scores = [np.sort(logits, axis=None)[::-1] for logits in all_preds]
            sorted_preds = [[(id, score) for id, score in zip(pred_id, pred_score)] for (pred_id, pred_score) in zip(sorted_pred_ids, sorted_pred_scores)]
            # np.savez(open(os.path.join(args.output_dir, 'pred_scores.npz'), 'w'), all_preds)
            json.dump([logits.squeeze().tolist() for logits in all_preds],
                      open(os.path.join(args.output_dir, 'pred_scores.json'), 'w'))
            json.dump(data_infos, open(os.path.join(args.output_dir, 'data_infos.json'), 'w'))
            write_selection_preds(eval_dataset.dataset_walker, args.output_file, data_infos, sorted_preds, topk=args.top_k)
    elif args.task.lower() == "detection":
        all_labels = np.concatenate(all_labels)
        all_pred_ids = np.concatenate([np.argmax(logits, axis=1) for logits in all_preds])
        # all_pred_ids = np.concatenate([(logits[:, 1] > 0.15).astype(int) for logits in all_preds]) # the prediction decision threshold can be tuned to balance between precision and recall
        accuracy = np.sum(all_pred_ids == all_labels) / len(all_labels)
        f1 = sklearn.metrics.f1_score(all_labels, all_pred_ids)
        precision = sklearn.metrics.precision_score(all_labels, all_pred_ids)
        recall = sklearn.metrics.recall_score(all_labels, all_pred_ids)
        result = {"loss": eval_loss, "accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}
        if args.output_file:
            np.save(open(os.path.join(args.output_dir, 'pred_scores.npy'), 'wb'), np.concatenate(all_preds))
            write_detection_preds(eval_dataset.dataset_walker, args.output_file, data_infos, all_pred_ids)
    elif args.task.lower() == "binary-classification":
        all_labels = np.expand_dims(np.concatenate(all_labels), axis=1)
        all_pred_ids = np.concatenate([(logits> 0.5).astype(int) for logits in all_preds])
        accuracy = np.sum(all_pred_ids == all_labels) / len(all_labels)
        f1 = sklearn.metrics.f1_score(all_labels, all_pred_ids)
        precision = sklearn.metrics.precision_score(all_labels, all_pred_ids)
        recall = sklearn.metrics.recall_score(all_labels, all_pred_ids)
        result = {"loss": eval_loss, "accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}
        if args.output_file:
            np.save(open(os.path.join(args.output_dir, 'pred_scores.npy'), 'wb'), np.concatenate(all_preds))
            write_detection_preds(eval_dataset.dataset_walker, args.output_file, data_infos, all_pred_ids)
    elif args.task.lower() in ['classification', 'domain-detection']:
        all_labels = np.concatenate(all_labels)
        all_pred_ids = np.concatenate([np.argmax(logits, axis=1) for logits in all_preds])

        accuracy = np.sum(all_pred_ids == all_labels) / len(all_labels)
        precision = sklearn.metrics.precision_score(all_labels, all_pred_ids, average='macro')
        recall = sklearn.metrics.recall_score(all_labels, all_pred_ids, average='macro')
        result = {"loss": eval_loss, "accuracy": accuracy, "precision": precision, "recall": recall}
        if hasattr(args, "output_dir"):
            np.save(open(os.path.join(args.output_dir, 'pred_scores.npy'), 'wb'), np.concatenate(all_preds))
            json.dump(all_pred_ids.tolist(), open(os.path.join(args.output_dir, f'preds-{args.task.lower()}.json'), 'w'))
    else:
        raise ValueError("args.task not in ['generation', 'selection', 'detection', \
                         'classification', 'domain-detection'], got %s" % args.task)

    if args.local_rank in [-1, 0]:
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results %s *****" % desc)
            writer.write("***** Eval results %s *****\n" % desc)
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--params_file", type=str, help="JSON configuration file")
    parser.add_argument("--model_for_ks", type=str, default='gpt2', help="Model used for knowledge selection")
    parser.add_argument("--eval_only", action="store_true",
                        help="Perform evaluation only")
    parser.add_argument("--resume_training", action="store_true",
                        help="Resume training from last checkpoint")
    parser.add_argument("--checkpoint", type=str, help="Saved checkpoint directory")
    parser.add_argument("--cache_dir", type=str, default="/home/ec2-user/models",
                        help="Path to store pre-trained model files.")
    parser.add_argument("--model_name_or_path", type=str, default="/home/ec2-user/models/gpt2-small",
                        help="Path to pre-trained model.")
    parser.add_argument("--dataset_dir", type=str,
                        help="Dir to the dataset for sequence classification (mainly for paraphrasing identification).")
    parser.add_argument("--per_gpu_eval_batch_size", type=int, default=1,
                        help="Maximum batch size for each gpu.")
    parser.add_argument("--num_train_epochs", type=int, default=5,
                        help="Number of training epochs.")
    parser.add_argument("--history_max_tokens", type=int, default=-1,
                        help="Maximum length in tokens for history, will override that value in config.")
    parser.add_argument("--knowledge_max_tokens", type=int, default=-1,
                        help="Maximum length in tokens for knowledge, will override that value in config.")
    parser.add_argument("--history_sent_num_for_ks", type=int, default=-1,
                        help="How many utterances in history dialogue are used for knowledge selection.")
    parser.add_argument("--n_candidates", type=int, default=5,
                        help="How many candidates used for knowledge selection training.")
    parser.add_argument("--top_k", type=int, default=5,
                        help="How many top selections are written into the result file.")
    parser.add_argument("--knowledge_selection_method", type=str, default='ans',
                        help="Which part of knowledge used for knowledge selection: question, answer, or both.")
    parser.add_argument("--negative_mix_percent", type=str, default='0.25,0.25,0.25,0.25',
                        help="Percent of different mixing strategy for negative sampling.")
    parser.add_argument("--use_tfidf_rate", type=float, default=0.5,
                        help="Percentage of cases where we would like to rank the negative samples using TF-IDF.")
    parser.add_argument("--use_hinge_loss", action="store_true",
                        help="Whether use hinge loss for knowledge selection")
    parser.add_argument("--dataroot", type=str, default="data",
                        help="Path to dataset.")
    parser.add_argument("--eval_dataroot", type=str, default="data",
                        help="Path to dataset for evaluation.")
    parser.add_argument("--knowledge_file", type=str, default="knowledge.json",
                        help="knowledge file name.")
    parser.add_argument("--eval_dataset", type=str, default="val",
                        help="Dataset to evaluate on, will load dataset from {dataroot}/{eval_dataset}")
    parser.add_argument("--no_labels", action="store_true",
                        help="Read a dataset without labels.json. This option is useful when running "
                             "knowledge-seeking turn detection on test dataset where labels.json is not available.")
    parser.add_argument("--labels_file", type=str, default=None,
                        help="If set, the labels will be loaded not from the default path, but from this file instead."
                        "This option is useful to take the outputs from the previous task in the pipe-lined evaluation.")
    parser.add_argument("--entities_file", type=str, default=None,
                        help="For pipeline methods, this file logs all detected entities.")
    parser.add_argument("--domain_list", type=str, default=None,
                        help="Specify the domains for train and test.")
    parser.add_argument("--output_dir", type=str, default="", help="Predictions will be written into this folder.")
    parser.add_argument("--output_file", type=str, default="", help="Predictions will be written to this file.")
    parser.add_argument("--negative_sample_method", type=str, choices=["all", "mix", "oracle"],
                        default="oracle", help="Negative sampling method for knowledge selection, will override the value in config.")
    parser.add_argument("--eval_all_snippets", action='store_true',
                        help="If set, the candidates to be selected would be all knowledge snippets, not sampled subset.")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Name of the experiment, checkpoints will be stored in runs/{exp_name}")
    parser.add_argument("--eval_desc", type=str, default="",
                        help="Optional description to be listed in eval_results.txt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for model initialization")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    verify_args(args, parser)

    # load args from params file and update the args Namespace
    with open(args.params_file, "r") as f:
        params = json.load(f)
        args = vars(args)

        update_additional_params(params, args)
        args.update(params)
        if 'gpt2' in args["model_for_ks"]:
            if 'gpt2' in params:
                args.update(params['gpt2'])
        else:
            args.update(params['bert'])
        args = Namespace(**args)
    
    args.params = params # used for saving checkpoints
    set_default_params(args)
    dataset_args = Namespace(**args.dataset_args)
    set_default_dataset_params(dataset_args)
    dataset_args.local_rank = args.local_rank
    dataset_args.task = args.task
    dataset_args.knowledge_selection_method = args.knowledge_selection_method
    dataset_args.history_sent_num_for_ks = args.history_sent_num_for_ks
    dataset_args.history_max_utterances = args.history_sent_num_for_ks
    dataset_args.model_for_ks = args.model_for_ks
    dataset_args.dataset_dir = args.dataset_dir
    dataset_args.negative_mix_percent = list(map(float, args.negative_mix_percent.split(',')))
    dataset_args.n_candidates = args.n_candidates
    dataset_args.use_tfidf_rate = args.use_tfidf_rate
    # dataset_args.history_max_tokens = args.history_max_tokens
    # dataset_args.knowledge_max_tokens = args.knowledge_max_tokens

    os.makedirs(args.output_dir, exist_ok=True)

    if args.domain_list is not None:
        args.domain_list = args.domain_list.split(',')


    # Setup CUDA, GPU & distributed training
    args.distributed = (args.local_rank != -1)
    if not args.distributed:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    args.n_gpu = torch.cuda.device_count() if not args.distributed else 1
    args.device = device

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    dataset_class, model_class, run_batch_fn_train, run_batch_fn_eval = get_classes(args.task, args.model_for_ks)

    if args.eval_only or args.resume_training:
        if not args.output_dir:
            args.output_dir = args.checkpoint
        model = model_class.from_pretrained(args.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    else:
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        # set output_past to False for DataParallel to work during evaluation
        config.output_past = False
        if args.task == 'classification':
            config.num_labels = len(processors[args.dataset_dir.split('/')[-1].lower()]().get_labels())
        elif args.task == 'domain-detection':
            config.num_labels = 3
        elif args.task == 'detection':
            config.num_labels = 2
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
        model = model_class.from_pretrained(args.model_name_or_path, config=config, cache_dir=args.cache_dir)
        model.resize_token_embeddings(len(tokenizer))

    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)
    if not args.eval_only:
        if args.domain_list is not None:
            train_dataset = dataset_class(dataset_args, tokenizer, split_type="train", domain_list=args.domain_list)
            eval_dataset = dataset_class(dataset_args, tokenizer, split_type="val", domain_list=args.domain_list)
        else:
            train_dataset = dataset_class(dataset_args, tokenizer, split_type="train")
            eval_dataset = dataset_class(dataset_args, tokenizer, split_type="val")

        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, tokenizer, run_batch_fn_train, run_batch_fn_eval)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
        if args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir, exist_ok=True)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
            with open(os.path.join(args.output_dir, "params.json"), "w") as jsonfile:
                json.dump(params, jsonfile, indent=2)

            # Load a trained model and vocabulary that you have fine-tuned
            model = model_class.from_pretrained(args.output_dir)
            tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
            model.to(args.device)

    # Evaluation
    result = {}
    if args.local_rank in [-1, 0]:
        dataset_args.dataroot = args.eval_dataroot
        if args.domain_list is not None:
            eval_dataset = dataset_class(dataset_args, tokenizer, split_type=args.eval_dataset, labels=not args.no_labels,
                                        labels_file=args.labels_file, entities_file=args.entities_file,
                                         domain_list=args.domain_list)
        else:
            eval_dataset = dataset_class(dataset_args, tokenizer, split_type=args.eval_dataset,
                                         labels=not args.no_labels,
                                         labels_file=args.labels_file, entities_file=args.entities_file)
        result = evaluate(args, eval_dataset, model, tokenizer, run_batch_fn_eval, desc=args.eval_desc or "val")

    return result


if __name__ == "__main__":
    main()
