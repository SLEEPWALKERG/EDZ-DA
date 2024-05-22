from multiprocessing import Pool
from tqdm import tqdm
from openai import OpenAI
from func_timeout import func_set_timeout
client = OpenAI(api_key="")  # input your openai key


@func_set_timeout(100)
def call_openai_api(input_text):
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": input_text}],
        temperature=0.0,  # do not forget to change the temperature.
    )
    return my_parse(response)


def process_text(inputs):
    for _ in range(3):
        try:
            result = call_openai_api(inputs['gpt_input'])
            # print(f"Output for input:\n{result}\n{'='*30}")
            return inputs['align_data'], result
        except:
            continue
    return None


def parallel_gpt_generate(inputs):
    # 设置最大并发任务数
    max_workers = 10  # 你可以根据需要调整这个数字

    with Pool(max_workers) as pool, tqdm(total=len(inputs)) as pbar:
        results = []
        for result in pool.imap_unordered(process_text, inputs):
            results.append(result)
            pbar.update(1)
    return results

    # with open('output.txt', 'w', encoding='utf-8') as file:
    #     for i, (input_text, generated_text) in enumerate(results):
    #         file.write(f"Input {i + 1}:\n{input_text}\nOutput {i + 1}:\n{generated_text}\n{'=' * 30}\n")


def my_parse(res):
    if res == '':
        return res
    else:
        return res.choices[0].message.content
