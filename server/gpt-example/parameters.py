from typing import Optional
from enum import Enum


class NextTokenChooserParameters(object):
    # TODO: remove hardcoded values and make them configurable
    def __init__(self):
        self.temperature = 1
        self.top_k = None
        self.top_p = 1
        self.typical_p = 1
        self.do_sample = None
        self.seed = 155956926070191748
        self.repetition_penalty = 1
        self.watermark = None


class StoppingCriteriaParameters(object):
    def __init__(self, max_new_tokens):
        self.ignore_eos_token = False
        self.max_new_tokens = max_new_tokens
        self.stop_sequences = []


class Request(object):
    def __init__(self, inputs, parameters, stopping_parameters) -> None:
        self.id = None
        self.inputs = inputs
        self.input_length = len(self.inputs.split(" "))
        self.parameters = parameters
        self.stopping_parameters = stopping_parameters
        self.truncate = 100


class Batch(object):
    def __init__(self, requests) -> None:
        self.id = None
        self.requests = requests
        self.size = len(self.requests)


class PrefillTokensParameters(object):
    def __init__(self, token_ids, logprobs, texts) -> None:
        self.token_ids = token_ids
        self.logprobs = logprobs
        self.texts = texts


class Generation(object):
    def __init__(self, request_id, prefill_tokens, token_id, token_logprob,
                 token_text, token_is_special, generated_text) -> None:
        self.request_id = request_id
        self.prefill_tokens = prefill_tokens
        self.token_id = token_id
        self.token_logprob = token_logprob
        self.token_text = token_text
        self.token_is_special = token_is_special
        self.generated_text = generated_text


class FinishReasonParameters(Enum):
    # number of generated tokens == `max_new_tokens`
    Length = "length"
    # the model generated its end of sequence token
    EndOfSequenceToken = "eos_token"
    # the model generated a text included in `stop_sequences`
    StopSequence = "stop_sequence"


class GeneratedText(object):
    def __init__(self, text, generated_tokens, finish_reason: FinishReasonParameters, seed: Optional[int]):
        self.text = text
        self.generated_tokens = generated_tokens
        self.finish_reason = finish_reason
        self.seed = seed