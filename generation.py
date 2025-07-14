
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import json
import os
import sys
import time
from pathlib import Path
from packaging import version
from typing import Callable, Generator, List, Optional
from transformers import StoppingCriteria, StoppingCriteriaList

import torch
import torch.nn.functional as F
from termcolor import cprint

from model.modeling_mllama import MllamaForConditionalGeneration
from model.processing_mllama import MllamaProcessor
from transformers import AutoProcessor
import torch.nn as nn
#from ..datatypes import GenerationResult, QuantizationMode, RawContent, RawMessage, ToolPromptFormat
#from .args import ModelArgs
#from .chat_format import ChatFormat, LLMInput
#from .model import Transformer
#from .multimodal.model import CrossAttentionTransformer
#from .tokenizer import Tokenizer

def split_tensor_before_sub(tensor, sublist):
    sub_tensor = torch.tensor(sublist, device=tensor.device)
    n = len(sub_tensor)

    for i in range(tensor.size(0) - n + 1):
        if torch.equal(tensor[i:i+n], sub_tensor):
            before = tensor[:i]      # PRIMA di sublist, sub non incluso
            sub = tensor[i:i+n]      # sublist
            after = tensor[i+n:]     # dopo sublist
            return before, sub, after
    return None, None, None  # sublist non trovato

def split_list_before_sub(big_list, sub_list):
    n = len(sub_list)

    for i in range(len(big_list) - n + 1):
        if big_list[i:i+n] == sub_list:
            before = big_list[:i]     # before sublist, sub not included
            sub = big_list[i:i+n]     # sublist itself
            after = big_list[i+n:]    # after sublist
            return before, sub, after
    return None, None, None  # sublist not found

def is_xccl_available():
    if version.parse(torch.__version__).release >= version.parse("2.7").release:
        return torch.distributed.distributed_c10d.is_xccl_available()
    return False

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


class Llama3_Vision(nn.Module):
    def __init__(self, llama_model="", config=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Loading LLAMA')
        self.llama_processor = MllamaProcessor.from_pretrained(llama_model)
        kwargs = {"torch_dtype": torch.float16}
        #if config is not None:
        #    kwargs["config"] = config
        self.llama_model = MllamaForConditionalGeneration.from_pretrained(llama_model, **kwargs)
        self.llama_model = self.llama_model.to(self.device)

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

    @torch.inference_mode()
    def generate(
        self,
        prompts,
        images,
        max_gen_tokens = 100,
        return_tokens =False,
    ):
        if len(prompts) != len(images):
            raise ValueError("The number of prompts and image sets must be the same.")

        input_texts = []
        for i, prompt in enumerate(prompts):
            input_text = self.llama_processor.apply_chat_template(prompt, add_generation_prompt=True)
            input_texts.append(input_text)
            print(f"Input text {i}: {input_text}")

        self.llama_processor.tokenizer.padding_side = 'left'
        tokens = self.llama_processor(
            images,
            input_texts,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_tensors="pt"
        ).to(self.device)
        # tokens['input_ids'] = (B, max_prompt_lenght)
        # tokens['attention_mask'] = (B, max_prompt_lenght)

        output = self.llama_model.generate(**tokens, max_new_tokens=max_gen_tokens) 
        
        prompt_len = tokens.input_ids.shape[-1]
        generated_ids = output[:, prompt_len:]  
     
        output_texts = self.llama_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
            add_special_tokens=False
        )

        outputs_texts = []
        for response, prompt in zip(output_texts, input_texts):
            response = response.strip() 
            outputs_texts.append(response)
            print(f"Response: {response}\n")

        return outputs_texts, generated_ids
    
    @torch.inference_mode()
    def generate_multimodal_with_attention_blocking(
        self,
        prompts,
        answer_tokens,
        images,
        thought_answer=None,
        temperature: float = 0.6,
        #top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        block_types: List[str] = None,
        k: int = 9,
        #logprobs: bool = False,
        #echo: bool = False,
        #print_model_input: bool = False,
        #logits_processor: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ): # -> Generator[List[GenerationResult], None, None]:
        
        input_texts = []
        if thought_answer is not None: 
            assert len(prompts) == len(thought_answer), "The number of prompts and cot sets must be the same."
        
        for i, prompt in enumerate(prompts):
            input_text = self.llama_processor.apply_chat_template(prompt, add_generation_prompt=True)
            if thought_answer is not None:
                input_text = input_text + thought_answer[i]
            input_texts.append(input_text)
            print(f"Input text {i}: {input_text}")

        params = self.llama_model.config
        if max_gen_len is None or max_gen_len == 0 or max_gen_len >= params.text_config.max_position_embeddings:
            max_gen_len = params.text_config.max_position_embeddings - 1

        bsz = len(input_texts)
        #assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        
        self.llama_processor.tokenizer.padding_side = 'left'
        processor_tokens = self.llama_processor(
            images,
            input_texts,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_tensors="pt"
        ).to(self.device)

        prompt_tokens = processor_tokens['input_ids'] #(batch_size, seq_len)
        #answer_tokens = [self.llama_processor.tokenizer.encode(x, bos=False, eos=False) for x in answers]
        answer_tokens = torch.tensor(answer_tokens, device=self.device).unsqueeze(0)  # (1, batch_size, n_tokens), unsqueeze to later expand it to n_layers

        #min_prompt_len = min(len(t) for t in prompt_tokens)
        #max_prompt_len = max(len(t) for t in prompt_tokens)

        #if max_prompt_len >= params.text_config.max_position_embeddings:
        #    cprint(f"Out of token budget {max_prompt_len} vs {params.text_config.max_position_embeddings}", "red")
        #    return

        #total_len = min(max_gen_len + max_prompt_len, params.text_config.max_position_embeddings)

        prompt_len = len(prompt_tokens[0])
        if prompt_len >= params.text_config.max_position_embeddings:
            cprint(f"Out of token budget {prompt_len} vs {params.text_config.max_position_embeddings}", "red")
            return
        total_len = min(max_gen_len + prompt_len, params.text_config.max_position_embeddings)

        pad_id = self.llama_processor.tokenizer.pad_token_id #128004
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.full((bsz, total_len), 0, dtype=torch.long, device=self.device)                  
        cross_attention_mask = torch.full((bsz, total_len, processor_tokens['cross_attention_mask'].shape[2], processor_tokens['cross_attention_mask'].shape[3]), 0, dtype=torch.long, device=self.device)
        
        for j, t in enumerate(prompt_tokens):
            tokens[j, : len(t)] = t.clone().detach()  
            attention_mask[j, : len(t)] = processor_tokens['attention_mask'][j].clone().detach()
            cross_attention_mask[j, : len(t)] = processor_tokens['cross_attention_mask'][j].clone().detach()
        
        prob_layers_type = {}
        if block_types is not None:
            for block_type in block_types:
                prev_pos = 0
                cur_pos = prompt_len

                if block_type in ['last_to_last', 'image_to_last', 'question_to_last', 'image_to_question', 'full_attention']:
                    # for each token in the answer
                    for i in range(answer_tokens.shape[2]):
                        answer_token = answer_tokens[:, :, i].unsqueeze(-1)                 #(1, batch_size, 1)
                        if answer_token.item() == self.llama_processor.tokenizer.eos_token_id : #128009 = <|eot_id|>
                            continue
                        logits = self.llama_model.forward_with_attention_blocking(
                            input_ids=tokens[:, prev_pos:cur_pos],                          #(batch_size, seq_len)
                            attention_mask=attention_mask[:, prev_pos:cur_pos],             #(batch_size, seq_len)
                            pixel_values=processor_tokens['pixel_values'],                  #(batch_size, max_num_images, max_num_tiles, channels, image_size, image_size)
                            aspect_ratio_ids=processor_tokens['aspect_ratio_ids'],          #(batch_size, max_num_images)
                            aspect_ratio_mask=processor_tokens['aspect_ratio_mask'],        #(batch_size, max_num_images, max_num_tiles)
                            cross_attention_mask=cross_attention_mask[:, prev_pos:cur_pos], #(batch_size, seq_len, max_num_images, max_num_tiles)
                            block_type=block_type,
                            k=k,
                            prompt_lengths=cur_pos
                        )                                                                   #(n_layers, batch_size, vocab_size)/(1, batch_size, vocab_size)

                        if block_type in ['last_to_last', 'image_to_last', 'question_to_last', 'image_to_question']:
                            probs = torch.softmax(logits / temperature if temperature > 0 else logits, dim=-1)  # (n_layers, batch_size, vocab_size)
                            indices = answer_token.expand_as(logits[..., :answer_token.shape[2]])               # (n_layers, batch_size, 1)
                            new_prob = probs.gather(dim=2, index=indices).squeeze(-1).transpose(0, 1)           # (n_layers, batch_size, 1) -> (batch_size, n_layers)
                            if block_type in prob_layers_type:
                                prob_layers_type[block_type] = prob_layers_type[block_type] * new_prob
                            else:
                                prob_layers_type[block_type] = new_prob 
                        elif block_type == 'full_attention':
                            probs = torch.softmax(logits.squeeze(0) / temperature if temperature > 0 else logits, dim=-1)   # (batch_size, vocab_size)
                            indices = answer_token.squeeze(0)                                                               # (batch_size, 1)
                            new_prob = probs.gather(dim=1, index=indices).squeeze(-1)                                       # (batch_size, 1) -> (batch_size)
                            
                            if block_type in prob_layers_type:
                                prob_layers_type[block_type] = prob_layers_type[block_type] * new_prob
                            else:
                                prob_layers_type[block_type] = new_prob
                        else:
                            raise NotImplementedError
                        
                        tokens[:, cur_pos] = answer_tokens[0, :, i]                                                         #(batch_size, tot_len)
                        attention_mask[:, cur_pos] = 1
                        cross_attention_mask[:, cur_pos, :, :] = cross_attention_mask[:, cur_pos-1, :, :]
                        cur_pos += 1
                        #prompt_lengths += 1
                else:
                    raise NotImplementedError
                
        print(list(prob_layers_type.keys()))
        return prob_layers_type 

def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token