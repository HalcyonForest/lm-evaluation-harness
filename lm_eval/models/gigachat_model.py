import os
from typing import Any, List, Tuple

from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import retry_on_specific_exceptions


eval_logger = utils.eval_logger


def gigachat_completion(
    client, #: gigachat.GigaChat,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    system: str,
    **kwargs,
) -> str:
    """Wrapper function around the Anthropic completion API client with exponential back-off
    in case of RateLimitError.
git 
    params:
        client: gigachat.GigaChat
            GigaChat API client
        model: str
            GigaChat model e.g. 'GigaChat-Pro', 'GigaChat'
        prompt: str
            Prompt to feed to the model
        max_tokens: int
            Maximum number of tokens to sample from the model
        temperature: float
            Sampling temperature
        system: str
            Instructions to gc
        kwargs: Any
            Additional model_args to pass to the API client. May be:
            profanity check: bool, turn onn censor. Default: False
            top_p: float, nucleus params
            repetition_penalty: float, repetition_penalty.
            For more: https://developers.sber.ru/docs/ru/gigachat/api/reference/rest/post-chat 
    """

    try:
        import gigachat
    except ModuleNotFoundError:
        raise Exception(
            "attempted to use 'gigachat' LM type, but package `gigachat` is not installed. \
please install gigachat via `pip install 'gigachat'`",
        )
    messages=[]
    if system:
        messages.append(
            gigachat.models.Messages(
                role=gigachat.models.MessagesRole.SYSTEM,
                content=system,
            )  
        )
        
    messages.append(
            gigachat.models.Messages(
                role=gigachat.models.MessagesRole.USER,
                content=prompt,
            )
        )

    def completion():
        
        payload = gigachat.models.Chat(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        response = client.chat(
            payload
        )
        return response.choices[0].message.content

    return completion()

@register_model("gigachat_llms")
class GigaChatLM(LM):

    def __init__(
        self,
        model: str = "GigaChat",
        max_tokens: int = 256,
        temperature: float = 1e-10,  
        **kwargs,  # top_p,  etc.
    ) -> None:
        """GigaChat API wrapper.

        :param model: str
            GC model e.g. 'GigaChat', 'GigaChar-Pro'
        :param max_tokens_to_sample: int
            Maximum number of tokens to sample from the model
        :param temperature: float
            Sampling temperature
        :param kwargs: Any
            Additional model_args to pass to the API client
        """
        super().__init__()

        try:
            import gigachat
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'gigachat' LM type, but package `gigachat` is not installed. \
please install gigachat via `pip install 'gigachat'`",
            )

        self.model = model
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        self.client = gigachat.GigaChat(
            credentials=os.environ.get("GIGACHAT_API_KEY"),
            scope="GIGACHAT_API_CORP",
            verify_ssl_certs=False
            )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs

    @property
    def eot_token_id(self):
        # Not sure but anthropic.HUMAN_PROMPT ?
        raise NotImplementedError("No idea about gc tokenization.")

    @property
    def max_length(self) -> int:
        return 2048

    @property
    def max_gen_toks(self) -> int:
        return self.max_tokens

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError("No support for logits.")

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError("No support for logits.")

    def tok_encode(self, string: str) -> List[int]:
        return NotImplementedError("No idea about gc tokenization.")

    def tok_decode(self, tokens: List[int]) -> str:
        return NotImplementedError("No idea about gc tokenization.")

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("No support for logits.")

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        try:
            import gigachat
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'gigachat' LM type, but package `gigachat` is not installed. \
please install gigachat via `pip install 'gigachat'`",
            )

        if not requests:
            return []

        _requests: List[Tuple[str, dict]] = [req.args for req in requests]

        res = []
        for request in tqdm(_requests, disable=disable_tqdm):
            try:
                inp = request[0]
                request_args = request[1]
                # generation_kwargs
                max_gen_toks = request_args.get("max_gen_toks", self.max_length)
                temperature = request_args.get("temperature", self.temperature)
                if temperature==0:
                    temperature=1e-10
                system = request_args.get("instruction", None)
                response = gigachat_completion(
                    client=self.client,
                    model=self.model,
                    prompt=inp,
                    max_tokens_to_sample=max_gen_toks,
                    temperature=temperature,  
                    system=system,
                    **self.kwargs,
                )
                res.append(response)

                self.cache_hook.add_partial("generate_until", request, response)
            except gigachat.exceptions.AuthenticationError as e: 
                eval_logger.critical(f"""API error {e.args[1]}: {e.args[2].decode('utf8').split('"message":')[-1][:-1]}""")
                break
            except gigachat.exceptions.ResponseError as e:
                eval_logger.critical(f"""API error {e.args[1]}: {e.args[2].decode('utf8').split('"message":')[-1][:-1]}""") 
                break

        return res

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override generate_until
        raise NotImplementedError()

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("No support for logits.")

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("No support for logits.")


