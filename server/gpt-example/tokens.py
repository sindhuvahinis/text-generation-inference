import re
import torch

from transformers import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    PreTrainedTokenizerBase,
)
from typing import List, Tuple, Optional

from parameters import StoppingCriteriaParameters, NextTokenChooserParameters



class Sampling:
    def __init__(self, seed: int, device: str = "cpu"):
        self.generator = torch.Generator(device)
        self.generator.manual_seed(seed)
        self.seed = seed

    def __call__(self, logits):
        probs = torch.nn.functional.softmax(logits, -1)
        next_tokens = torch.multinomial(probs, num_samples=1, generator=self.generator)
        return next_tokens


class Greedy:
    def __call__(self, logits):
        return logits.argmax()


class NextTokenChooser:
    def __init__(
            self,
            parameters: NextTokenChooserParameters,
            device="cpu"
    ):
        self.warpers = LogitsProcessorList()
        self.choice = Sampling(parameters.seed, device) if parameters.do_sample else Greedy()

    def __call__(self, input_ids, scores):
        # Warp logits
        if scores.shape[0] > 1:
            # only warp the last token logits
            scores[-1:, :] = self.warpers(input_ids, scores[-1:, :])
        else:
            scores = self.warpers(input_ids, scores)

        # Compute logprobs
        logprobs = torch.log_softmax(scores, -1)

        # Choose tokens
        next_id = self.choice(scores[-1])

        return next_id.view(1, 1), logprobs


class StopSequenceCriteria:
    def __init__(self, stop_sequence: str):
        stop_sequence = re.escape(stop_sequence)
        self.regex = re.compile(f".*{stop_sequence}$")

    def __call__(self, output: str) -> bool:
        if self.regex.findall(output):
            return True
        return False


class StoppingCriteria:
    def __init__(
            self,
            parameters: StoppingCriteriaParameters,
            tokenizer: PreTrainedTokenizerBase,
    ):

        stop_sequence_criterias = [
            StopSequenceCriteria(sequence) for sequence in parameters.stop_sequences
        ]

        self.eos_token_id = tokenizer.eos_token_id
        self.stop_sequence_criterias = stop_sequence_criterias
        self.max_new_tokens = parameters.max_new_tokens
        self.current_tokens = 0
        self.current_output = ""
        self.ignore_eos_token = parameters.ignore_eos_token

    def __call__(self, last_token: int, last_output: str) -> Tuple[bool, Optional[str]]:
        self.current_tokens += 1
        if self.current_tokens >= self.max_new_tokens:
            return True, 0

        if not self.ignore_eos_token and last_token == self.eos_token_id:
            return True, ""

        self.current_output += last_output
        for stop_sequence_criteria in self.stop_sequence_criterias:
            if stop_sequence_criteria(self.current_output):
                return True, ""

        return False, None