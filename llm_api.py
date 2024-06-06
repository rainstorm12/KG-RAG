import random
from http import HTTPStatus
import dashscope
import time
#llm api调用
def call_stream_with_qwen(prompt):
    messages = [
        {"role": "user", "content": prompt}]
    responses = dashscope.Generation.call(
        'qwen1.5-110b-chat',
        # 'qwen1.5-32b-chat',
        # 'qwen1.5-1.8b-chat',#free
        messages=messages,
        seed=random.randint(1, 10000),  # set the random seed, optional, default to 1234 if not set
        result_format='message',  # set the result to be "message"  format.
        stream=True,
        output_in_full=True  # get streaming output incrementally
    )
    full_content = ''
    past_key_values, history = None, []
    for response in responses:
        if response.status_code == HTTPStatus.OK:
            message = response.output.choices[0]['message']['content']
            # print(message[len(full_content):], end="",flush=True)
            full_content = message
            yield prompt, full_content, past_key_values, history
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
            time.sleep(5)
    # print('\nFull content: \n',full_content)
    # return full_content

#llm api调用
def call_with_qwen(prompt):
    messages = [
        {"role": "user", "content": prompt}]
    responses = dashscope.Generation.call(
        'qwen1.5-110b-chat',
        # 'qwen1.5-32b-chat',
        # 'qwen1.5-1.8b-chat',#free
        messages=messages,
        seed=random.randint(1, 10000),  # set the random seed, optional, default to 1234 if not set
        result_format='message',  # set the result to be "message"  format.
        stream=True,
        output_in_full=True  # get streaming output incrementally
    )
    full_content = ''
    for response in responses:
        if response.status_code == HTTPStatus.OK:
            message = response.output.choices[0]['message']['content']
            print(message[len(full_content):], end="",flush=True)
            full_content = message
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
            time.sleep(5)
    # print('\nFull content: \n',full_content)
    return full_content

def glm_init():
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b-int4", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm2-6b-int4", trust_remote_code=True).half().cuda()
    model = model.eval()
    return model,tokenizer

def call_stream_with_glm(query,model,tokenizer,past_key_values, history):
    current_length = 0
    for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history,
                                                            past_key_values=past_key_values,
                                                            return_past_key_values=True):
        # print(response[current_length:], end="", flush=True)
        # current_length = len(response)
        yield query, response, past_key_values, history

def call_with_glm(query,model,tokenizer):
    past_key_values, history = None, []
    current_length = 0
    for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history,
                                                            past_key_values=past_key_values,
                                                            return_past_key_values=True):
        print(response[current_length:], end="", flush=True)
        current_length = len(response)
    return response

def llm_init(llm):
    if llm=='qwen':
        return None,None
    elif llm=='glm':
        model,tokenizer = glm_init()
        return model,tokenizer

def call_stream_with_messages(prompt,llm='qwen',model=None,tokenizer=None,past_key_values=None, history=[]):
    if llm=='qwen':
        yield from call_stream_with_qwen(prompt)
    elif llm=='glm':
        yield from call_stream_with_glm(prompt,model,tokenizer,past_key_values, history)
        
def call_with_messages(prompt,llm='qwen',model=None,tokenizer=None):
    if llm=='qwen':
        return call_with_qwen(prompt)
    elif llm=='glm':
        return call_with_glm(prompt,model,tokenizer)       