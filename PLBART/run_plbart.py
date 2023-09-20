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
Text to code generation pipeline in CodeXGLUE
"""

from __future__ import absolute_import, division, print_function

import argparse
from cmath import log
import glob
from json import encoder
import logging
import os
import pickle
import random
import re
import shutil
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from dataset import CodeChangeDataset, read_data
from beam import Beam
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from bleu import _bleu
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
                          PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer,
                          MBartConfig, MBartForConditionalGeneration, MBartTokenizer)

from model import PLBART_Model

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    'PLBart': (PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer),
    'MBart': (MBartConfig, MBartForConditionalGeneration, MBartTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def update_config(model, tokenizer):
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

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
    # print(filename)
    examples = []
    assert len(filename.split(',')) == 3
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    location_filename = filename.split(',')[2]
    idx = 0
    with open(src_filename, encoding='utf-8') as f1, open(trg_filename, encoding='utf-8') as f2, open(location_filename, encoding='utf-8') as f3:
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
        source_tokens = tokenizer.tokenize(example.source)[:args.source_length - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length
        location = example.location

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.target_length - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.target_length - len(target_ids)
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


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

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


# def train(args, train_dataset, model, tokenizer, fh, pool):
def train(args, model, tokenizer, fh, pool):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        args.tensorboard_dir = os.path.join(args.output_dir, 'tensorboard')
        if not os.path.exists(args.tensorboard_dir):
            os.makedirs(args.tensorboard_dir)
        tb_writer = SummaryWriter(args.tensorboard_dir)

    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_filename = args.train_file
    # train_filename = args.data_dir + '/train' + '.buggy-fixed.buggy' +','+ args.data_dir + '/train' + '.buggy-fixed.fixed'
    train_examples = read_examples(train_filename)
    train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')
    all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
    all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
    all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)
    all_locations = torch.tensor([f.location for f in train_features], dtype=torch.long)
    train_dataset = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask, all_locations)

    train_sampler = RandomSampler(train_dataset)

    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, drop_last=True)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, drop_last=True, num_workers=8)
    total_examples = len(train_dataset) * (
        torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    batch_size = args.batch_size * args.gradient_accumulation_steps * (
        torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    # if args.max_steps > 0:
    #     t_total = args.max_steps
    #     args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    if args.num_train_epochs > 0:
        t_total = total_examples // batch_size * args.num_train_epochs
    args.max_steps = t_total
    model.to(args.device)
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last, map_location="cpu"))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last, map_location="cpu"))
    if args.local_rank == 0:
        torch.distributed.barrier()
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank % args.gpu_per_node],
                                                          output_device=args.local_rank % args.gpu_per_node,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", total_examples)
    logger.info("  Num epoch = %f", t_total * batch_size / total_examples)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb = 0.0, 0.0, 0.0, 0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    best_bleu = 0.0
    patience_counter = 0
    best_top1 = 0.0
    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        logger.info("Starting Epoch : %d" % idx)
        _train_dataloader = tqdm(train_dataloader,
                                 total=int(np.ceil(len(train_dataset)/args.batch_size)),
                                 desc='', leave=True)
        # for step, (batch, token_labels) in enumerate(_train_dataloader):
        # dev_bleu, dev_EM, top1, top5 = eval_bleu(args, model, tokenizer, file_type='eval')
        for step, batch in enumerate(_train_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            # inputs = batch.to(args.device)
            source_ids, source_mask, target_ids, target_mask, locations = batch
            # attn_mask = torch.tensor(token_labels.clone().detach() != 0, dtype=torch.uint8, device=args.device)
            # loss_mask = torch.tensor(token_labels.clone().detach() == 2, dtype=torch.uint8, device=args.device)
            model.train()
            # outputs = model(inputs, attention_mask=attn_mask)
            # tgt_attn_mask = -1e4 * (1 - self.bias[:target_ids.shape[1], :target_ids.shape[1]])
            if args.localization:
                outputs, location_logits = model(input_ids=source_ids,
                                                 attention_mask=source_mask, 
                                                 decoder_input_ids=target_ids, 
                                                 decoder_attention_mask=target_mask)
            else:
                outputs = model(input_ids=source_ids,
                                attention_mask=source_mask, 
                                decoder_input_ids=target_ids, 
                                decoder_attention_mask=target_mask)
            logits = outputs[0]
            labels = target_ids
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_mask = target_mask[..., 1:].ne(0).view(-1) == 1
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            flatten_shift_loss_mask = loss_mask[..., :-1].contiguous().view(-1)
            ids = torch.nonzero(flatten_shift_loss_mask).view(-1)
            edit_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[ids], shift_labels.view(-1)[ids])

            if args.localization:
                # print('=================================')
                # print(location_logits.shape)
                # print(locations.shape)
                # print('=================================')
                loc_loss = smooth_CE_loss(args.max_lines, location_logits, locations, eps=args.smooth_eps)
                if args.only_loc:
                    loss = loc_loss
                else:
                    loss = edit_loss + args.lam * loc_loss
            else:
                loss = edit_loss

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                _train_dataloader.set_description(desc="   Epoch: %d  steps: %s  ppl: %s" % (idx, global_step, round(avg_loss, 5)))
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss
                    tr_nb = global_step
                    pass
                pass
            if args.max_steps > 0 and global_step > args.max_steps:
                print('max_steps: {}'.format(args.max_steps))
                print('global_step: {}'.format(global_step))
                break
                pass
            pass

        # Start Saving After Every epoch
        if args.localization and args.only_loc:
            dev_bleu, dev_EM, top1, top5 = eval_bleu(args, model, tokenizer, file_type='eval')
        else:
            dev_bleu, dev_EM, top1, top5 = eval_bleu(args, model, tokenizer, file_type='eval', num=1000)
        logger.info(f"dev bleu: {dev_bleu}, dev EM: {dev_EM}, top1: {top1}, top5: {top5}")
        output_dir = args.output_dir
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        if args.localization and args.only_loc:
            if top1 > best_top1:
                best_top1 = top1
                logger.info(f"best top1 updated. saved in {output_dir}")
                logger.info(f"best top1: {best_top1}")

                patience_counter = 0
                best_output_dir = os.path.join(args.output_dir, 'checkpoint-best')
                if not os.path.exists(best_output_dir):
                    os.makedirs(best_output_dir)
                logger.info("Saving BEST model checkpoint to %s", best_output_dir)
                torch.save(model_to_save.state_dict(), os.path.join(best_output_dir, "pytorch_model.bin"))
                # model_to_save.save_pretrained(best_output_dir)
                config_to_save = model_to_save.backbone_config()
                config_to_save.save_pretrained(best_output_dir)
                tokenizer.save_pretrained(best_output_dir)
                torch.save(args, os.path.join(best_output_dir, "training_args.bin"))
            else:
                patience_counter += 1
        
        else:
            if dev_bleu > best_bleu:
                best_bleu = dev_bleu
                logger.info(f"best bleu updated. saved in {output_dir}")
                logger.info(f"best bleu: {best_bleu}")
                patience_counter = 0
                best_output_dir = os.path.join(args.output_dir, 'checkpoint-best')
                if not os.path.exists(best_output_dir):
                    os.makedirs(best_output_dir)
                logger.info("Saving BEST model checkpoint to %s", best_output_dir)
                torch.save(model_to_save.state_dict(), os.path.join(best_output_dir, "pytorch_model.bin"))
                # model_to_save.save_pretrained(best_output_dir)
                config_to_save = model_to_save.backbone_config()
                config_to_save.save_pretrained(best_output_dir)
                tokenizer.save_pretrained(best_output_dir)
                torch.save(args, os.path.join(best_output_dir, "training_args.bin"))
            else:
                patience_counter += 1

        last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
        logger.info("Saving last model checkpoint to %s", output_dir)
        if not os.path.exists(last_output_dir):
            os.makedirs(last_output_dir)
        torch.save(model_to_save.state_dict(), os.path.join(last_output_dir, "pytorch_model.bin"))
        # model_to_save.save_pretrained(last_output_dir)
        config_to_save = model_to_save.backbone_config()
        config_to_save.save_pretrained(last_output_dir)
        tokenizer.save_pretrained(last_output_dir)
        idx_file = os.path.join(last_output_dir, 'idx_file.txt')
        with open(idx_file, 'w', encoding='utf-8') as idxf:
            idxf.write(str(0) + '\n')
        torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(last_output_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", last_output_dir)
        step_file = os.path.join(last_output_dir, 'step_file.txt')
        with open(step_file, 'w', encoding='utf-8') as stepf:
            stepf.write(str(global_step) + '\n')
        if args.max_steps > 0 and global_step > args.max_steps:
            break
        if patience_counter >= args.max_patience:
            logger.info('Reached maximum patience, Exiting!')
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def eval_bleu(args, model, tokenizer, file_type='test', num=99999999):
    if file_type == 'eval':
        test_filename = args.dev_file
    elif file_type == 'test':
        test_filename = args.test_file
    # test_filename = args.data_dir + '/{}'.format(file_type) + '.buggy-fixed.buggy' +','+ args.data_dir + '/{}'.format(file_type) + '.buggy-fixed.fixed'
    test_examples = read_examples(test_filename)
    if file_type == 'eval' and num < len(test_examples):
        test_examples = random.sample(test_examples, min(1000, len(test_examples)))
    test_features = convert_examples_to_features(test_examples, tokenizer, args, stage=file_type)
    all_source_ids = torch.tensor([f.source_ids for f in test_features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in test_features], dtype=torch.long)
    all_target_ids = torch.tensor([f.target_ids for f in test_features], dtype=torch.long)
    all_target_mask = torch.tensor([f.target_mask for f in test_features], dtype=torch.long)
    all_locations = torch.tensor([f.location for f in test_features], dtype=torch.long)
    test_dataset = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask, all_locations)

    test_sampler = SequentialSampler(test_dataset)
    # test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, drop_last=True)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1)

    model.to(args.device)
    model.zero_grad()
    model.eval()
    preds = []
    # for step, batch in enumerate(_train_dataloader):
    # for step, (batch, token_labels) in enumerate(tqdm(test_dataloader, total=min(num, len(test_dataset)))):

    all_location_logits = torch.Tensor([])
    all_locations = torch.tensor([], dtype=torch.long)
    for step, batch in enumerate(tqdm(test_dataloader)):
        if step >= num:
            break
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, source_mask, target_ids, target_mask, locations = batch
        # inputs = batch.to(args.device)
        with torch.no_grad():
            beam_size = args.beam_size
            m = torch.nn.LogSoftmax(dim=-1)

            ###############################
            input_ids = source_ids.repeat((beam_size, 1))
            attention_mask = source_mask.repeat((beam_size, 1))

            # encoder = model.get_encoder()
            # encoder_outputs = encoder(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask
            # )
            if args.localization:
                # print('====================================')
                encoder_outputs, location_logits = model.get_encoder_out(input_ids=input_ids, attention_mask=attention_mask)
                location_logits = location_logits[0].unsqueeze(0)
                # print('encoder_outputs shape: {}'.format(location_logits.shape))
                # print('locations shape: {}'.format(locations.shape))
                location_logits = F.softmax(location_logits, dim=-1)
                all_location_logits = torch.cat((all_location_logits, location_logits.detach().cpu()), dim=0)
                # locations = locations.detach().cpu()
                all_locations = torch.cat((all_locations, locations.detach().cpu()), dim=0)
                if args.only_loc:
                    continue
            else:
                encoder_outputs = model.get_encoder_out(input_ids=input_ids, attention_mask=attention_mask)
            ###############################


            p = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(source_ids.shape[0]):
                beam = Beam(beam_size, tokenizer.bos_token_id, tokenizer.eos_token_id)
                decoder_input_ids = beam.getCurrentState()
                # past_key_values = None
                for _ in range(args.target_length):
                    if beam.done():
                        break

                    # input_ids = source_ids.repeat((decoder_input_ids.shape[0], 1))
                    # attention_mask = source_mask.repeat((decoder_input_ids.shape[0], 1))

                    ###############################
                    
                    transformer_outputs = model(
                        encoder_outputs=encoder_outputs,
                        # past_key_values=past_key_values,
                        decoder_input_ids=decoder_input_ids,
                        attention_mask=attention_mask
                    )

                    # past_key_values = transformer_outputs.past_key_values


                    ###############################


                    # transformer_outputs = model(
                    #     input_ids=input_ids,
                    #     attention_mask=attention_mask,
                    #     decoder_input_ids=decoder_input_ids
                    #     )

                    out = m(transformer_outputs[0][:, -1, :]).data
                    beam.advance(out)

                    # decoder_input_ids = beam.getCurrentState()

                    decoder_input_ids.data.copy_(decoder_input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    decoder_input_ids = torch.cat((decoder_input_ids, beam.getCurrentState()), -1)

                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:beam_size]

                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (args.target_length - len(p))).view(1, -1) for p in pred]
                p.append(torch.cat(pred, 0).unsqueeze(0))
            p = torch.cat(p, 0)
            for pred in p:
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                preds.append(text)
    
    if args.localization:
        top1, top5 = top_1_5_acc(all_location_logits, all_locations)
        if args.only_loc:
            return None, None, top1, top5
    else:
        top1, top5 = None, None

    golds = []
    
    for gold in test_examples:
        golds.append(gold.target)

    # datas = read_data(data_dir=args.data_dir, file_type=file_type)
    # for (src, tgt) in datas[:num]:
    #     golds.append(tgt)

    assert len(preds) == len(golds), 'Pred %d\tGold %d' %(len(preds), len(golds))

    EM = []
    with open(os.path.join(args.output_dir, f"{file_type}.output"), 'w', encoding='utf-8') as f, open(
            os.path.join(args.output_dir, f"{file_type}.gold"), 'w', encoding='utf-8') as f1:
        for pred, gold in zip(preds, golds):
            f.write(pred + '\n')
            f1.write(gold + '\n')
            EM.append(pred.split() == gold.split())

    bleu_score = round(
        _bleu(os.path.join(args.output_dir, f"{file_type}.gold"), os.path.join(args.output_dir, f"{file_type}.output")),
        2)
    EM = round(np.mean(EM) * 100, 2)
    return bleu_score, EM, top1, top5


def main():
    parser = argparse.ArgumentParser()

    # ## Required parameters
    # parser.add_argument("--data_dir", default=None, type=str, required=True,
    #                     help="The input data path.")

    parser.add_argument("--train_file", default=None, type=str,
                        help="The train filenames (source and target files).")
    parser.add_argument("--dev_file", default=None, type=str,
                        help="The dev filename. (source and target files).")
    parser.add_argument("--test_file", default=None, type=str,
                        help="The test filename. (source and target files).")

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--model_type", default="PLBart", type=str,
                        help="The model architecture to be fine-tuned.")
    # parser.add_argument("--model_type", default="MBart", type=str,
    #                     help="The model architecture to be fine-tuned.")

    parser.add_argument("--pretrain_dir", default="", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--config_dir", type=str,
                        help="config name. Required when training from scratch")
    parser.add_argument("--tokenizer_dir", type=str,
                        help="Pre-trained tokenizer dir. Required when training from scratch")
    parser.add_argument("--load_name", type=str, default="pretrained",
                        help="Load pretrained model name")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    # parser.add_argument("--block_size", default=512, type=int,
    #                     help="Optional input sequence length after tokenization."
    #                          "The training dataset will be truncated in block of this size for training."
    #                          "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--source_length", default=512, type=int)
    parser.add_argument("--target_length", default=256, type=int)
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_infer", action='store_true',
                        help="Whether to run inference on test set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    # parser.add_argument("--warmup_steps", default=0, type=int,
    #                     help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=10,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=-1,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=1234,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--node_index", type=int, default=-1,
                        help="node index if multi-node running")
    parser.add_argument("--gpu_per_node", type=int, default=-1,
                        help="num of gpus per node")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    parser.add_argument('--log_file', type=str, default='')
    parser.add_argument('--tensorboard_dir', type=str)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--max_patience', type=int, default=15)

    parser.add_argument('--localization', type=boolean_string, default=False)
    parser.add_argument('--max_lines', type=int, default=2)
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--smooth_eps', type=float, default=0.0)
    parser.add_argument('--only_loc', type=boolean_string, default=False)

    parser.add_argument('--load_model_path', type=str, default='')

    pool = None
    args = parser.parse_args()

    # args.output_dir = os.path.join(args.output_dir, args.dataset)

    if args.model_type in ["bert", "roberta", "distilbert"] and not args.mlm:
        raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
                         "flag (masked language modeling).")

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    logger.warning(
        "local_rank: %d, node_index: %d, gpu_per_node: %d" % (args.local_rank, args.node_index, args.gpu_per_node))
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.local_rank += args.node_index * args.gpu_per_node
        args.n_gpu = 1
    args.device = device
    # args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s, world size: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16,
        torch.distributed.get_world_size() if args.local_rank != -1 else 1)

    # 使用FileHandler输出到文件
    fh = logging.FileHandler(args.log_file)
    logger.addHandler(fh)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if args.do_train and os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.pretrain_dir = os.path.join(checkpoint_last)
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    # Load pre-trained model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    pretrained = args.pretrain_dir
    logger.info(pretrained)
    if pretrained:
        # tokenizer = tokenizer_class.from_pretrained(pretrained, do_lower_case=args.do_lower_case, bos_token='<s>',
        #                                             eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>',
        #                                             sep_token='<s>')
        # print()
        # print(pretrained)
        # print()
        # logger.info(pretrained)
        tokenizer = tokenizer_class.from_pretrained(pretrained, do_lower_case=args.do_lower_case)
        logger.info(tokenizer.encode("<s> hello world <pad> </s>"))

        config = config_class.from_pretrained(pretrained)
        # if args.do_train:
        backbone = model_class.from_pretrained(pretrained, config=config)
        # else:
        #     backbone = model_class(config)
        # config = backbone.config
        # model = model_class.from_pretrained('../models/pretrained/checkpoints/pytorch_model.bin', config=config)
        # model.resize_token_embeddings(len(tokenizer))
        # update_config(model, tokenizer)
        logger.info(backbone.config)
    else:
        # tokenizer = tokenizer_class.from_pretrained(args.tokenizer_dir, bos_token='<s>', eos_token='</s>',
        #                                             pad_token='<pad>', unk_token='<|UNKNOWN|>',
        #                                             sep_token='<s>')
        tokenizer = tokenizer_class.from_pretrained(pretrained, do_lower_case=args.do_lower_case)
        
        # args.vocab_size = tokenizer.vocab_size
        config = config_class.from_pretrained(args.config_dir)
        backbone = model_class(config)
        # model.resize_token_embeddings(len(tokenizer))
        # update_config(model, tokenizer)
    
    model = PLBART_Model(args=args, backbone=backbone, config=backbone.config)

    model_parameters = model.parameters()
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info(f"Model has a total of {num_params} trainable parameters")

    if args.do_infer and args.load_model_path:
        logger.info("Load parameters from {} for testing".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        # train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        # global_step, tr_loss = train(args, train_dataset, model, tokenizer, fh, pool)

        global_step, tr_loss = train(args, model, tokenizer, fh, pool)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # dev_bleu, dev_EM = eval_bleu(args, model, tokenizer, file_type='eval', num=100)
        # print(f"dev bleu: {dev_bleu}, dev EM: {dev_EM}")

    if args.do_eval:  # only works on 1 GPU
        dev_bleu, dev_EM, top1, top5 = eval_bleu(args, model, tokenizer, file_type='eval')
        logger.info(f"dev bleu: {dev_bleu}, dev EM: {dev_EM}, top1: {top1}, top5: {top5}")
        print(f"dev bleu: {dev_bleu}, dev EM: {dev_EM}, top1: {top1}, top5: {top5}")

    if args.do_infer:  # only works on 1 GPU
        test_bleu, test_EM, top1, top5 = eval_bleu(args, model, tokenizer, file_type='test')
        logger.info(f"dev bleu: {test_bleu}, dev EM: {test_EM}, top1: {top1}, top5: {top5}")
        print(f"dev bleu: {test_bleu}, dev EM: {test_EM}, top1: {top1}, top5: {top5}")


if __name__ == "__main__":
    main()
