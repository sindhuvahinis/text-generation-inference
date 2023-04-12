from text_generation_server.pb.generate_pb2 import FinishReason
from typing import Optional

class NextTokenChooserParametersPB(object):
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


class StoppingCriteriaParametersPB(object):
    def __init__(self, max_new_tokens):
        self.ignore_eos_token = False
        self.max_new_tokens = max_new_tokens
        self.stop_sequences = []


class RequestPB(object):
    def __init__(self, inputs, parameters, stopping_parameters) -> None:
        self.id = None
        self.inputs = inputs
        self.input_length = len(self.inputs.split(" "))
        self.parameters = parameters
        self.stopping_parameters = stopping_parameters
        self.truncate = 100


class BatchPB(object):
    def __init__(self, requests) -> None:
        self.id = None
        self.requests = requests
        self.size = len(self.requests)


class PrefillTokensPB(object):
    def __init__(self, token_ids, logprobs, texts) -> None:
        self.token_ids = token_ids
        self.logprobs = logprobs
        self.texts = texts


class GenerationPB(object):
    def __init__(self, request_id, prefill_tokens, token_id, token_logprob,
                 token_text, token_is_special, generated_text) -> None:
        self.request_id = request_id
        self.prefill_tokens = prefill_tokens
        self.token_id = token_id
        self.token_logprob = token_logprob
        self.token_text = token_text
        self.token_is_special = token_is_special
        self.generated_text = generated_text

class GeneratedTextPB(object):
    def __init__(self, text, generated_tokens, finish_reason: FinishReason, seed: Optional[int]):
        self.text = text
        self.generated_tokens = generated_tokens
        self.finish_reason = finish_reason
        self.seed = seed