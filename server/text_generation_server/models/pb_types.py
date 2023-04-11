## classes to mock types defined in generate.proto
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
        self.max_new_tokens = max_new_tokens
        self.stop_sequences = []

class RequestPB(object):
    def __init__(self, inputs, parameters, stopping_parameters) -> None:
        self.id = None
        self.inputs = inputs
        self.input_length = len(self.inputs.split(" "))
        self.parameters = parameters
        self.stopping_parameters = stopping_parameters

class BatchPB(object):
    def __init__(self, requests) -> None:
        self.id = None
        self.requests = requests
        self.size = len(self.requests)