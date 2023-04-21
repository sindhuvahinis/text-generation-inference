# from typing import Optional
# from enum import Enum
#
#
# class NextTokenChooserParameters(object):
#     # TODO: remove hardcoded values and make them configurable
#     def __init__(self):
#         self.temperature = 1
#         self.top_k = None
#         self.top_p = 1
#         self.typical_p = 1
#         self.do_sample = None
#         self.seed = 155956926070191748
#         self.repetition_penalty = 1.0
#         self.watermark = None
#
#
# class StoppingCriteriaParameters(object):
#     def __init__(self, max_new_tokens):
#         self.ignore_eos_token = False
#         self.max_new_tokens = max_new_tokens
#         self.stop_sequences = []
#
#
# class Request(object):
#     def __init__(self, inputs, parameters, stopping_parameters) -> None:
#         self.id = None
#         self.inputs = inputs
#         self.input_length = len(self.inputs.split(" "))
#         self.parameters = parameters
#         self.stopping_parameters = stopping_parameters
#         self.truncate = 100
#
#
# class Batch(object):
#     def __init__(self, requests) -> None:
#         self.id = None
#         self.requests = requests
#         self.size = len(self.requests)
#
#
# class PrefillTokens(object):
#     def __init__(self, token_ids, logprobs, texts) -> None:
#         self.token_ids = token_ids
#         self.logprobs = logprobs
#         self.texts = texts
#
#
# class Generation(object):
#     def __init__(self, request_id, prefill_tokens, token_id, token_logprob,
#                  token_text, token_is_special, generated_text) -> None:
#         self.request_id = request_id
#         self.prefill_tokens = prefill_tokens
#         self.token_id = token_id
#         self.token_logprob = token_logprob
#         self.token_text = token_text
#         self.token_is_special = token_is_special
#         self.generated_text = generated_text
#
#
# class FinishReasonParameters(Enum):
#     # number of generated tokens == `max_new_tokens`
#     Length = "length"
#     # the model generated its end of sequence token
#     EndOfSequenceToken = "eos_token"
#     # the model generated a text included in `stop_sequences`
#     StopSequence = "stop_sequence"
#
#
# class GeneratedText(object):
#     def __init__(self, text, generated_tokens, finish_reason: FinishReasonParameters, seed: Optional[int]):
#         self.text = text
#         self.generated_tokens = generated_tokens
#         self.finish_reason = finish_reason
#         self.seed = seed

import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from transformers import PreTrainedTokenizerBase

from text_generation_server.pb import generate_pb2
from text_generation_server.pb.generate_pb2 import FinishReason


class Batch(ABC):
    @abstractmethod
    def to_pb(self) -> generate_pb2.Batch:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_pb(
            cls,
            pb: generate_pb2.Batch,
            tokenizer: PreTrainedTokenizerBase,
            device: torch.device,
    ) -> "Batch":
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def concatenate(cls, batches: List["Batch"]) -> "Batch":
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError


@dataclass
class GeneratedText:
    text: str
    generated_tokens: int
    finish_reason: FinishReason
    seed: Optional[int]

    def to_pb(self) -> generate_pb2.GeneratedText:
        return generate_pb2.GeneratedText(
            text=self.text,
            generated_tokens=self.generated_tokens,
            finish_reason=self.finish_reason,
            seed=self.seed,
        )


@dataclass
class PrefillTokens:
    token_ids: List[int]
    logprobs: List[float]
    texts: List[str]

    def to_pb(self) -> generate_pb2.PrefillTokens:
        return generate_pb2.PrefillTokens(
            ids=self.token_ids, logprobs=self.logprobs, texts=self.texts
        )

    def __len__(self):
        return len(self.token_ids)


@dataclass
class Generation:
    request_id: int
    prefill_tokens: Optional[PrefillTokens]
    token_id: int
    token_logprob: float
    token_text: str
    token_is_special: bool
    generated_text: Optional[GeneratedText]

    def to_pb(self) -> generate_pb2.Generation:
        return generate_pb2.Generation(
            request_id=self.request_id,
            prefill_tokens=self.prefill_tokens.to_pb()
            if self.prefill_tokens is not None
            else None,
            token_id=self.token_id,
            token_logprob=self.token_logprob,
            token_text=self.token_text,
            token_is_special=self.token_is_special,
            generated_text=self.generated_text.to_pb()
            if self.generated_text is not None
            else None,
        )
