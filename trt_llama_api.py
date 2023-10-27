# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import flask
from flask import jsonify

from transformers import AutoTokenizer
import gc
import json
import torch
import numpy as np
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
import tensorrt_llm
from pathlib import Path
import uuid
import time
from typing import Any, Callable, Optional, Dict
from utils import EOS

EOS_TOKEN = 2
PAD_TOKEN = 2

DEFAULT_CONTEXT_WINDOW = 3900
DEFAULT_NUM_OUTPUTS = 256

try:
    from pydantic.v1 import (
        BaseModel,
        Field,
        PrivateAttr,
        root_validator,
        validator,
        create_model,
        StrictFloat,
        StrictInt,
        StrictStr,
    )
    from pydantic.v1.fields import FieldInfo
    from pydantic.v1.error_wrappers import ValidationError
except ImportError:
    from pydantic import (
        BaseModel,
        Field,
        PrivateAttr,
        root_validator,
        validator,
        create_model,
        StrictFloat,
        StrictInt,
        StrictStr,
    )
    from pydantic.fields import FieldInfo
    from pydantic.error_wrappers import ValidationError


def make_resData(data, chat=False, promptToken=[]):
    resData = {
        "id": f"chatcmpl-{str(uuid.uuid4())}" if (chat) else f"cmpl-{str(uuid.uuid4())}",
        "object": "chat.completion" if (chat) else "text_completion",
        "created": int(time.time()),
        "truncated": data["truncated"],
        "model": "LLaMA",
        "usage": {
            "prompt_tokens": data["prompt_tokens"],
            "completion_tokens": data["completion_tokens"],
            "total_tokens": data["prompt_tokens"] + data["completion_tokens"]
        }
    }
    if (len(promptToken) != 0):
        resData["promptToken"] = promptToken
    if (chat):
        # only one choice is supported
        resData["choices"] = [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": data["content"],
            },
            "finish_reason": "stop" if data["stopped"] else "length"
        }]
    else:
        # only one choice is supported
        resData["choices"] = [{
            "text": data["content"],
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop" if data["stopped"] else "length"
        }]
    return resData


def make_resData_stream(data, chat=False, start=False):
    resData = {
        "id": "chatcmpl" if (chat) else "cmpl",
        "object": "chat.completion.chunk" if (chat) else "text_completion.chunk",
        "created": int(time.time()),
        "model": "LLaMA",
        "choices": [
            {
                "finish_reason": None,
                "index": 0
            }
        ]
    }
    slot_id = data["slot_id"]
    if (chat):
        if (start):
            resData["choices"][0]["delta"] = {
                "role": "assistant"
            }
        else:
            resData["choices"][0]["delta"] = {
                "content": data["content"]
            }
            if (data["stop"]):
                resData["choices"][0]["finish_reason"] = "stop" if data["stopped"]  else "length"
    else:
        resData["choices"][0]["text"] = data["content"]
        if (data["stop"]):
            resData["choices"][0]["finish_reason"] = "stop" if data["stopped"] else "length"

    return resData


class LLMMetadata(BaseModel):
    """LLM metadata."""

    context_window: int = DEFAULT_CONTEXT_WINDOW
    num_output: int = DEFAULT_NUM_OUTPUTS
    is_chat_model: bool = False
    is_function_calling_model: bool = False
    # By default we don't know the model name. We can set it automatically for
    # some types, but custom predictors (like locally loaded models) we won't
    # know.
    # Used for tests, logging, and sanity checks
    model_name: str = "unknown"


class TrtLlmAPI(BaseModel):
    model_path: Optional[str] = Field(
        description="The path to the trt engine."
    )
    temperature: float = Field(description="The temperature to use for sampling.")
    max_new_tokens: int = Field(description="The maximum number of tokens to generate.")
    context_window: int = Field(
        description="The maximum number of context tokens for the model."
    )
    messages_to_prompt: Callable = Field(
        description="The function to convert messages to a prompt.", exclude=True
    )
    completion_to_prompt: Callable = Field(
        description="The function to convert a completion to a prompt.", exclude=True
    )
    generate_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Kwargs used for generation."
    )
    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Kwargs used for model initialization."
    )
    verbose: bool = Field(description="Whether to print verbose output.")

    _model: Any = PrivateAttr()
    _model_config: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _max_new_tokens = PrivateAttr()
    _sampling_config = PrivateAttr()
    _verbose = PrivateAttr()

    def __init__(
            self,
            model_path: Optional[str] = None,
            engine_name: Optional[str] = None,
            tokenizer_dir: Optional[str] = None,
            temperature: float = 0.1,
            max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
            context_window: int = DEFAULT_CONTEXT_WINDOW,
            messages_to_prompt: Optional[Callable] = None,
            completion_to_prompt: Optional[Callable] = None,
            generate_kwargs: Optional[Dict[str, Any]] = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
            verbose: bool = False
    ) -> None:

        model_kwargs = model_kwargs or {}
        model_kwargs.update({"n_ctx": context_window, "verbose": verbose})
        self._max_new_tokens = max_new_tokens
        self._verbose = verbose
        # check if model is cached
        if model_path is not None:
            if not os.path.exists(model_path):
                raise ValueError(
                    "Provided model path does not exist. "
                    "Please check the path or provide a model_url to download."
                )
            else:
                engine_dir = model_path
                engine_dir_path = Path(engine_dir)
                config_path = engine_dir_path / 'config.json'

                # config function
                with open(config_path, 'r') as f:
                    config = json.load(f)
                use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
                remove_input_padding = config['plugin_config']['remove_input_padding']
                tp_size = config['builder_config']['tensor_parallel']
                pp_size = config['builder_config']['pipeline_parallel']
                world_size = tp_size * pp_size
                assert world_size == tensorrt_llm.mpi_world_size(), \
                    f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
                num_heads = config['builder_config']['num_heads'] // tp_size
                hidden_size = config['builder_config']['hidden_size'] // tp_size
                vocab_size = config['builder_config']['vocab_size']
                num_layers = config['builder_config']['num_layers']
                num_kv_heads = config['builder_config'].get('num_kv_heads', num_heads)
                paged_kv_cache = config['plugin_config']['paged_kv_cache']
                if config['builder_config'].get('multi_query_mode', False):
                    tensorrt_llm.logger.warning(
                        "`multi_query_mode` config is deprecated. Please rebuild the engine."
                    )
                    num_kv_heads = 1
                num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size

                self._model_config = ModelConfig(num_heads=num_heads,
                                                 num_kv_heads=num_kv_heads,
                                                 hidden_size=hidden_size,
                                                 vocab_size=vocab_size,
                                                 num_layers=num_layers,
                                                 gpt_attention_plugin=use_gpt_attention_plugin,
                                                 paged_kv_cache=paged_kv_cache,
                                                 remove_input_padding=remove_input_padding)

                assert pp_size == 1, 'Python runtime does not support pipeline parallelism'
                world_size = tp_size * pp_size

                runtime_rank = tensorrt_llm.mpi_rank()
                runtime_mapping = tensorrt_llm.Mapping(world_size,
                                                       runtime_rank,
                                                       tp_size=tp_size,
                                                       pp_size=pp_size)
                torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)
                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, legacy=False)
                self._sampling_config = SamplingConfig(end_id=EOS_TOKEN,
                                                       pad_id=PAD_TOKEN,
                                                       num_beams=1,
                                                       temperature=temperature)

                serialize_path = engine_dir_path / engine_name
                with open(serialize_path, 'rb') as f:
                    engine_buffer = f.read()
                decoder = tensorrt_llm.runtime.GenerationSession(self._model_config,
                                                                 engine_buffer,
                                                                 runtime_mapping,
                                                                 debug_mode=False)
                self._model = decoder
        messages_to_prompt = messages_to_prompt or generic_messages_to_prompt
        completion_to_prompt = completion_to_prompt or (lambda x: x)

        generate_kwargs = generate_kwargs or {}
        generate_kwargs.update(
            {"temperature": temperature, "max_tokens": max_new_tokens}
        )
        super().__init__(
            model_path=model_path,
            temperature=temperature,
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            generate_kwargs=generate_kwargs,
            model_kwargs=model_kwargs,
            verbose=verbose,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "TrtLlmAPI"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=self.model_path,
        )

    def chat_complete(self, prompt: str, **kwargs: Any) -> flask.Response:
        return self.complete_common(prompt, True)

    def complete(self, prompt: str, **kwargs: Any) -> flask.Response:
        return self.complete_common(prompt, False)

    def complete_common(self, prompt: str, chat: bool, **kwargs: Any):
        assert len(prompt) > 0
        is_formatted = kwargs.pop("formatted", False)
        temperature = kwargs.pop("temperature", 1.0)
        #TODO: need to respect (truncate output after) stop strings.
        stop_strings = kwargs.pop("stop_strings", "")
        if not is_formatted:
            prompt = self.completion_to_prompt(prompt)

        input_text = prompt
        input_ids, input_lengths = self.parse_input(input_text, self._tokenizer,
                                                    EOS_TOKEN,
                                                    self._model_config)

        max_input_length = torch.max(input_lengths).item()
        self._model.setup(input_lengths.size(0), max_input_length, self._max_new_tokens, 1)  # beam size is set to 1
        if self._verbose:
            start_time = time.time()

        self._sampling_config.temperature = temperature
        output_ids = self._model.decode(input_ids, input_lengths, self._sampling_config)
        torch.cuda.synchronize()

        elapsed_time = None
        if self._verbose:
            end_time = time.time()
            elapsed_time = end_time - start_time

        output_txt, output_token_ids = self.get_output(output_ids,
                                                       input_lengths,
                                                       self._max_new_tokens,
                                                       self._tokenizer)

        if self._verbose:
            print(f"Input context length  : {input_ids.shape[1]}")
            print(f"Inference time        : {elapsed_time:.2f} seconds")
            print(f"Output context length : {len(output_token_ids)} ")
            print(f"Inference token/sec   : {(len(output_token_ids) / elapsed_time):2f}")

        # call garbage collected after inference
        torch.cuda.empty_cache()
        gc.collect()

        thisdict = dict(truncated=False,
                        prompt_tokens=input_ids.shape[1],
                        completion_tokens=len(output_token_ids),
                        content=str(output_txt),
                        stopped=False,
                        slot_id=1,
                        stop=True)

        resData = make_resData(thisdict, chat=chat)
        return jsonify(resData)

    def parse_input(self, input_text: str, tokenizer, end_id: int,
                    remove_input_padding: bool):
        input_tokens = []

        input_tokens.append(
            tokenizer.encode(input_text, add_special_tokens=False))

        input_lengths = torch.tensor([len(x) for x in input_tokens],
                                     dtype=torch.int32,
                                     device='cuda')
        if remove_input_padding:
            input_ids = np.concatenate(input_tokens)
            input_ids = torch.tensor(input_ids, dtype=torch.int32,
                                     device='cuda').unsqueeze(0)
        else:
            input_ids = torch.nested.to_padded_tensor(
                torch.nested.nested_tensor(input_tokens, dtype=torch.int32),
                end_id).cuda()

        return input_ids, input_lengths

    def remove_extra_eos_ids(self, outputs):
        outputs.reverse()
        while outputs and outputs[0] == 2:
            outputs.pop(0)
        outputs.reverse()
        outputs.append(2)
        return outputs

    def get_output(self, output_ids, input_lengths, max_output_len, tokenizer):
        num_beams = output_ids.size(1)
        output_text = ""
        outputs = None
        for b in range(input_lengths.size(0)):
            for beam in range(num_beams):
                output_begin = input_lengths[b]
                output_end = input_lengths[b] + max_output_len
                outputs = output_ids[b][beam][output_begin:output_end].tolist()
                outputs = self.remove_extra_eos_ids(outputs)
                output_text = tokenizer.decode(outputs)

        return output_text, outputs

    def stream_complete(self, prompt: str, **kwargs: Any) -> flask.Response:
        return self.stream_complete_common(prompt, False)

    def stream_chat_complete(self, prompt: str, **kwargs: Any) -> flask.Response:
        return self.stream_complete_common(prompt, True)

    def stream_complete_common(self, prompt: str, chat: bool, **kwargs: Any) -> flask.Response:
        assert len(prompt) > 0
        is_formatted = kwargs.pop("formatted", False)
        temperature = kwargs.pop("temperature", 1.0)
        stop_strings = kwargs.pop("stop_strings", "")
        if not is_formatted:
            prompt = self.completion_to_prompt(prompt)

        input_text = prompt
        input_ids, input_lengths = self.parse_input(input_text, self._tokenizer,
                                                    EOS_TOKEN,
                                                    self._model_config)

        max_input_length = torch.max(input_lengths).item()
        self._model.setup(input_lengths.size(0), max_input_length, self._max_new_tokens, 1)  # beam size is set to 1
        self._sampling_config.temperature = temperature
        output_ids = self._model.decode(input_ids, input_lengths, self._sampling_config, streaming=True)

        def gen() -> flask.Response:
            thisdict = dict(truncated=False,
                            prompt_tokens=max_input_length,
                            completion_tokens=0,
                            content="",
                            stopped=False,
                            slot_id=1,
                            stop=False)
            resData = make_resData_stream(thisdict, chat=chat, start=True)
            yield 'data: {}\n'.format(json.dumps(resData))

            text = ""
            dictForDelta = dict(truncated=False,
                                prompt_tokens=max_input_length,
                                completion_tokens=0,
                                content="",
                                stopped=False,
                                slot_id=1,
                                stop=False)

            for output_ids_delta in output_ids:
                output_txt, output_token_ids = self.get_output(output_ids_delta,
                                                               input_lengths,
                                                               self._max_new_tokens,
                                                               self._tokenizer)

                if not dictForDelta["truncated"]:
                    delta_text = output_txt[len(text):]
                    text = output_txt.removesuffix(EOS)

                    dictForDelta["content"] = delta_text.removesuffix(EOS)
                    dictForDelta["completion_tokens"] = len(output_token_ids)
                    resData = make_resData_stream(dictForDelta, chat=chat)
                    yield 'data: {}\n'.format(json.dumps(resData))

                    for stop_string in stop_strings:
                        if stop_string in text:
                            dictForDelta["truncated"] = True
                            break


            # close last message
            dictForDelta["content"] = ""
            dictForDelta["stop"] = True
            resData = make_resData_stream(dictForDelta, chat=chat)
            yield 'data: {}\n'.format(json.dumps(resData))

        return flask.Response(gen(), mimetype='text/event-stream')
