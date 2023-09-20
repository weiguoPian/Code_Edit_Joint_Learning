# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import

import argparse
import logging
import os
import random
from io import open
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

from bleu import _bleu
from model import Seq2Seq

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

# def smooth_CE_loss(num_classes, logits, label, eps=0.1):
#     targets = F.one_hot(label, num_classes=num_classes)

#     if eps != 0.0:
#         targets = (targets * (1 - eps) + eps / num_classes)
#     loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=-1) * targets, dim=1))

#     return loss

def smooth_CE_loss(num_classes, logits, label, eps=0.1):
    if eps != 0.0:
        targets = torch.zeros_like(logits)
        targets.fill_(eps / (num_classes - 1))
        targets.scatter_(1, label.unsqueeze(1), 1.0 - eps)
    else:
        targets = F.one_hot(label, num_classes=num_classes)

    loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=-1) * targets, dim=1))

    return loss

def top_1_5_acc(logits, target):
    top1_res = logits.argmax(dim=1)
    _, top5_res = logits.topk(5, dim=1, largest=True, sorted=True)
    top1_acc = torch.eq(target, top1_res).sum().float() / len(target)
    top5_acc = torch.eq(target.view(-1, 1), top5_res).sum().float() / len(target)
    return top1_acc.item(), top5_acc.item()

class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 location,
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.location = location

def read_examples(filename):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 3
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    location_filename = filename.split(',')[2]
    idx = 0
    with open(src_filename) as f1, open(trg_filename) as f2, open(location_filename) as f3:
        for line1, line2, line3 in zip(f1, f2, f3):
            examples.append(
                Example(
                    idx=idx,
                    source=line1.strip(),
                    target=line2.strip(),
                    location=int(line3.strip()),
                )
            )
            idx += 1
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
                 location,
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask
        self.location = location


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        # source
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length
        location = example.location

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                source_mask,
                target_mask,
                location,
            )
        )
    return features


def _truncate_seq_pair(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--tokenizer_name", default="", required=True,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filenames (source and target files).")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. (source and target files).")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. (source and target files).")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")

    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--localization', type=boolean_string, default=False)
    parser.add_argument('--max_lines', type=int, default=2)
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--smooth_eps', type=float, default=0.0)
    parser.add_argument('--only_loc', type=boolean_string, default=True)

    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)

    # budild model
    encoder = model_class.from_pretrained(args.model_name_or_path, config=config)
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = Seq2Seq(args, encoder=encoder, decoder=decoder, config=config,
                    beam_size=args.beam_size, max_length=args.max_target_length,
                    sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    num_params = count_parameters(model)
    print('=' * 100)
    print('Total parameters : %d' % num_params)
    print('=' * 100)

    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        # Prepare training data loader
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)
        all_locations = torch.tensor([f.location for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask, all_locations)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=args.train_batch_size // args.gradient_accumulation_steps)
        num_train_optimization_steps = args.train_steps * args.gradient_accumulation_steps
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        # Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", num_train_optimization_steps * args.train_batch_size // (
                    len(train_examples) * args.gradient_accumulation_steps))

        model.train()
        dev_dataset = {}
        nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss, best_acc = 0, 0, 0, 0, 0, 1e6, 0
        bar = range(num_train_optimization_steps)
        train_dataloader = cycle(train_dataloader)
        eval_flag = True
        logger.info('smooth_eps = {}'.format(args.smooth_eps))
        # all_hidden_loc = []
        for _ in tqdm(bar):
            # if nb_tr_steps == 1000:
            #     break
            batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)
            source_ids, source_mask, target_ids, target_mask, locations = batch
            if args.localization:
                # loss1, _, _, location_logits = model(source_ids=source_ids, source_mask=source_mask,
                #                                      target_ids=target_ids, target_mask=target_mask)
                location_logits = model(source_ids=source_ids, source_mask=source_mask,
                                        target_ids=target_ids, target_mask=target_mask)
                loc_loss = smooth_CE_loss(args.max_lines, location_logits, locations, eps=args.smooth_eps)
                # loss = loss1 + args.lam * loc_loss
                loss = loc_loss
                # all_hidden_loc.append(hidden_loc.detach().cpu().numpy())
            else:
                loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                   target_ids=target_ids, target_mask=target_mask)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()
            train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)

            nb_tr_examples += source_ids.size(0)
            nb_tr_steps += 1
            loss.backward()

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                eval_flag = True

            if args.do_eval and ((global_step + 1) % args.eval_steps == 0) and eval_flag:
                # Eval model with dev dataset
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                eval_flag = False
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='dev')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                    all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
                    all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)
                    all_locations = torch.tensor([f.location for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask, all_locations)
                    dev_dataset['dev_loss'] = eval_examples, eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                # Start Evaling model
                model.eval()
                
                all_location_logits = torch.Tensor([])
                all_locations = torch.tensor([], dtype=torch.long)
                for batch in tqdm(eval_dataloader):
                    batch = tuple(t.to(device) for t in batch)
                    source_ids, source_mask, target_ids, target_mask, locations = batch
                    all_locations = torch.cat((all_locations, locations.detach().cpu()), dim=0)

                    with torch.no_grad():
                        location_logits = model(source_ids=source_ids, source_mask=source_mask,
                                                target_ids=target_ids, target_mask=target_mask)
                        location_logits = F.softmax(location_logits, dim=-1)
                        all_location_logits = torch.cat((all_location_logits, location_logits.detach().cpu()), dim=0)
                # Pring loss of dev dataset
                model.train()
                # if args.localization:
                top1, top5 = top_1_5_acc(all_location_logits, all_locations)
                result = {'top1': top1,
                          'top5': top5,
                          'global_step': global_step + 1,
                          'train_loss': round(train_loss, 5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)

                # save last checkpoint
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                if top1 > best_acc:
                    logger.info("  Best top1:%s", round(top1, 5))
                    logger.info("  " + "*" * 20)
                    best_acc = top1
                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-acc')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
        # all_hidden_loc = np.array(all_hidden_loc)
        # np.save('./save/all_hidden.npy', all_hidden_loc)             

    if args.do_test:
        files = []
        if args.dev_filename is not None:
            files.append(args.dev_filename)
        if args.test_filename is not None:
            files.append(args.test_filename)
        for idx, file in enumerate(files):
            logger.info("Test file: {}".format(file))
            eval_examples = read_examples(file)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
            all_locations = torch.tensor([f.location for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_source_ids, all_source_mask, all_locations)

            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            # p = []
            all_location_logits = torch.Tensor([])
            all_locations = torch.tensor([], dtype=torch.long)
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                source_ids, source_mask, locations = batch
                with torch.no_grad():
                    all_locations = torch.cat((all_locations, locations.detach().cpu()), dim=0)

                    location_logits = model(source_ids=source_ids, source_mask=source_mask)
                    location_logits = F.softmax(location_logits, dim=-1)
                    all_location_logits = torch.cat((all_location_logits, location_logits.detach().cpu()), dim=0)
            top1, top5 = top_1_5_acc(all_location_logits, all_locations)
            # model.train()
            # predictions = []
            # accs = []
            # with open(os.path.join(args.output_dir, "test_{}.output".format(str(idx))), 'w') as f, open(
            #         os.path.join(args.output_dir, "test_{}.gold".format(str(idx))), 'w') as f1:
            #     for ref, gold in zip(p, eval_examples):
            #         predictions.append(str(gold.idx) + '\t' + ref)
            #         f.write(ref + '\n')
            #         f1.write(gold.target + '\n')
            #         accs.append(ref == gold.target)
            # dev_bleu = round(_bleu(os.path.join(args.output_dir, "test_{}.gold".format(str(idx))).format(file),
            #                        os.path.join(args.output_dir, "test_{}.output".format(str(idx))).format(file)), 2)
            # logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
            # logger.info("  %s = %s " % ("xMatch", str(round(np.mean(accs) * 100, 4))))
            logger.info("  %s = %s, %s = %s " % ("top1_loc", str(top1), "top5_loc", str(top5)))
            logger.info("  " + "*" * 20)


if __name__ == "__main__":
    main()
