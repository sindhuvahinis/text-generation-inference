import sys

sys.path.append('/tmp/ws/text-generation-inference/server')

from parameters import Request, Batch, \
    StoppingCriteriaParameters, NextTokenChooserParameters
from causal_lm import CausalLM, CausalLMBatch
from gpt_neox import GPTNeoxSharded

import os
import argparse

INPUT_TEXT = "Amazon is"


def handle(args):
    if args.sharded:
        model = GPTNeoxSharded('EleutherAI/gpt-neox-20b', None, quantize=False)
    else:
        model = CausalLM('EleutherAI/gpt-neox-20b', None, quantize=False)
    print("loaded model")

    parameters = NextTokenChooserParameters()
    stopping_criteria = StoppingCriteriaParameters(args.max_new_tokens)
    request = Request(INPUT_TEXT, parameters, stopping_criteria)
    request_batch = CausalLMBatch([request])

    batch = model.batch_type.get_batch(request_batch, model.tokenizer, model.device)
    token_count = 1
    rank = int(os.getenv("RANK", "0"))
    while batch:
        generations, batch = model.generate_token(batch)
        if rank == 0:
            print(f"{token_count}: {generations[0].token_text}")
        token_count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--sharded", action="store_true")
    args = parser.parse_args()
    handle(args)