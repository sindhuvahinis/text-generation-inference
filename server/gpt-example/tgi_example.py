import sys

sys.path.append('/tmp/ws/text-generation-inference/server')

from pb_types import RequestPB, BatchPB, \
    StoppingCriteriaParametersPB, NextTokenChooserParametersPB
from causal_lm import CausalLM

import os

INPUT_TEXT = "Amazon is"


def handle(args):
    model = CausalLM('EleutherAI/gpt-neox-20b', None, quantize=False)
    print("loaded model")

    parameters = NextTokenChooserParametersPB()
    stopping_criteria = StoppingCriteriaParametersPB(args.max_new_tokens)
    request = RequestPB(INPUT_TEXT, parameters, stopping_criteria)
    request_batch = BatchPB([request])

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
    parser.add_argument("--max_new_tokens", type=int, default=20)
    args = parser.parse_args()
    handle(args)