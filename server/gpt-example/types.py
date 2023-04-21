
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


from parameters import FinishReasonParameters


class Batch(ABC):

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
    finish_reason: FinishReasonParameters
    seed: Optional[int]


@dataclass
class PrefillTokens:
    token_ids: List[int]
    logprobs: List[float]
    texts: List[str]

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
