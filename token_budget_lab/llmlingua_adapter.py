"""Optional LLMLingua-2 adapter. No model is loaded by the offline default."""
class LLMLingua2:
    def __init__(self, model, revision, device='cpu'):
        import re
        if not revision or not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('pin a model revision for reproducibility')
        from llmlingua import PromptCompressor
        self.compressor = PromptCompressor(model_name=model, use_llmlingua2=True,
                                           device_map=device, model_config={'revision': revision, 'trust_remote_code': False})

    def compress(self, query, context, budget, counter):
        # Question is deliberately not compressed. LLMLingua-2 is task-agnostic.
        if counter.count(context) <= budget:
            return context
        if budget == 0:
            return ''
        result = self.compressor.compress_prompt(context, rate=budget / counter.count(context))
        text = result['compressed_prompt']
        # Its internal tokenizer differs from our configured counter. Enforce the
        # same outer budget; expose this post-fit variant explicitly in reports.
        if counter.count(text) <= budget:
            return text
        while text and counter.count(text) > budget:
            text = text[:-1]
        return text
