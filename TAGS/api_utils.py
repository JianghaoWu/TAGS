import os
import time
import random
from wrapt_timeout_decorator import timeout
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def init_client(service="gptapi"):
    if service == "together":
        return OpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=os.getenv("TOGETHER_API_KEY")
        )
    elif service == "local":
        return OpenAI(
            base_url="http://localhost:4000/v1",  # vllm 默认地址
            api_key="EMPTY" 
        )
    else:
        raise NotImplementedError(f"Unknown client service: {service}")

# for API, using together:
client = init_client("together")
# for local vllm, uncomment the following line:
# client = init_client("local")


def generate_response_multiagent(engine, temperature, frequency_penalty, presence_penalty, stop, system_role, user_input):
    print("Generating response for engine: ", engine)
    start_time = time.time()
    response = client.chat.completions.create(
        model=engine,
        temperature=temperature,
        top_p=1,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        messages=[  
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_input}
        ],
    )
    end_time = time.time()
    print('Finish!')
    print("Time taken: ", end_time - start_time)

    return response, response.usage


def generate_response(engine, temperature, frequency_penalty, presence_penalty, stop, input_text):
    print("Generating response for engine: ", engine)
    start_time = time.time()
    response = client.chat.completions.create(
        model=engine,
        temperature=temperature,
        top_p=1,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        messages=[{"role": "user", "content": input_text}],
    )
    end_time = time.time()
    print('Finish!')
    print("Time taken: ", end_time - start_time)

    return response, response.usage

def generate_response_ins(engine, temperature, frequency_penalty, presence_penalty, stop, input_text, suffix, echo):
    print("Generating response for engine: ", engine)
    start_time = time.time()
    response = client.chat.completions.create(
        model=engine,
        prompt=input_text,
        temperature=temperature,
        top_p=1,
        suffix=suffix,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        echo=echo,
        logprobs=1,
    )
    end_time = time.time()
    print('Finish!')
    print("Time taken: ", end_time - start_time)

    return response, response.usage

class api_handler:
    def __init__(self, model):
        self.model = model

        if self.model == 'gpt-3.5':
            self.engine = 'gpt-3.5-turbo'
        elif self.model == 'gpt-4o':
            self.engine = 'gpt-4o'
        elif self.model == 'gpt-4o-mini':
            self.engine = 'gpt-4o-mini'
        elif self.model == 'deepseek-r1':
            self.engine = 'deepseek-r1'
        elif self.model == 'deepseek-v3':
            self.engine = 'deepseek-v3'
        elif self.model == 'claude-3-5-sonnet':
            self.engine = 'claude-3-5-sonnet'
        elif self.model == 'gpt-4':
            self.engine = 'gpt-4'
        elif self.model == 'Qwen2.5-7B-Instruct':
            self.engine = 'Qwen2.5-7B-Instruct'
        elif self.model == 'HuatuoGPT-o1-7B':
            self.engine = 'HuatuoGPT-o1-7B'
        elif self.model == 'HuatuoGPT-o1-8B':
            self.engine = 'HuatuoGPT-o1-8B'
        elif self.model == 'JSL-MedLlama-3-8B-v1.0':
            self.engine = 'JSL-MedLlama-3-8B-v1.0'
        elif self.model == 'JSL-MedLlama-3-8B-v2.0':
            self.engine = 'JSL-MedLlama-3-8B-v2.0'
        elif self.model == 'Llama3-OpenBioLLM-8B':
            self.engine = 'Llama3-OpenBioLLM-8B'
        elif self.model == 'Qwen2.5-32B-Instruct':
            self.engine = 'Qwen2.5-32B-Instruct'
        elif self.model == 'Meta-Llama-3-8B':
            self.engine = 'Meta-Llama-3-8B'
        else:
            raise NotImplementedError


    def get_output_multiagent(self, system_role, user_input, temperature=0,
                    frequency_penalty=0, presence_penalty=0, stop=None):
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response, usage = generate_response_multiagent(self.engine, temperature, frequency_penalty, presence_penalty, stop, system_role, user_input)
                if response.choices:
                    return response.choices[0].message.content, usage
                else:
                    return "ERROR.", None
            except Exception as error:
                print(f'Attempt {attempt+1} of {max_attempts} failed with error: {error}')
                if attempt == max_attempts - 1:
                    return "ERROR.", None


    def get_output(self, input_text, max_tokens, temperature=0,
                   suffix=None, stop=None, do_tunc=False, echo=False, ban_pronoun=False,
                   frequency_penalty=0, presence_penalty=0, return_prob=False):
        try:
            response, usage = generate_response(self.engine, temperature, frequency_penalty, presence_penalty, stop, input_text)
        except Exception as error:
            print("Timeout")
            try:
                response, usage = generate_response(self.engine, temperature, frequency_penalty, presence_penalty, stop, input_text)
            except Exception as error:
                print("Timeout occurred again. Exiting.")
                response = "ERROR."
                return response, None
        if response.choices:
            x = response.choices[0].message.content
        else:
            x = "ERROR."
            return x, None

        if do_tunc:
            y = x.strip()
            if '\n' in y:
                pos = y.find('\n')
                y = y[:pos]
            if 'Q:' in y:
                pos = y.find('Q:')
                y = y[:pos]
            if 'Question:' in y:
                pos = y.find('Question:')
                y = y[:pos]
            assert not ('\n' in y)
            if not return_prob:
                return y, usage

        if not return_prob:
            return x, usage

        output_token_offset_real, output_token_tokens_real, output_token_probs_real = [], [], []
        return x, (output_token_offset_real, output_token_tokens_real, output_token_probs_real), usage

"""
(Pdb) x
' Academy Award because The Curious Case of Benjamin Button won three Academy Awards, which are given by the Academy of Motion Picture Arts and Sciences.'
(Pdb) output_token_offset_real
[0, 8, 14, 22, 26, 34, 39, 42, 51, 58, 62, 68, 76, 83, 84, 90, 94, 100, 103, 107, 115, 118, 125, 133, 138, 142, 151]
(Pdb) output_token_tokens_real
[' Academy', ' Award', ' because', ' The', ' Curious', ' Case', ' of', ' Benjamin', ' Button', ' won', ' three', ' Academy', ' Awards', ',', ' which', ' are', ' given', ' by', ' the', ' Academy', ' of', ' Motion', ' Picture', ' Arts', ' and', ' Sciences', '.']
(Pdb) output_token_probs_real
[-0.7266144, -0.68505085, -0.044669915, -0.00023392851, -0.0021017971, -2.1768952e-05, -1.1430258e-06, -6.827632e-08, -3.01145e-05, -1.2231317e-05, -0.07086051, -2.7967804e-05, -6.6619094e-07, -0.41155097, -0.0020535963, -0.0021325003, -0.6671403, -0.51776046, -0.00014945272, -0.41470888, -3.076318e-07, -3.583558e-05, -2.9311614e-06, -3.869565e-05, -1.1430258e-06, -9.606849e-06, -0.017712338]
"""