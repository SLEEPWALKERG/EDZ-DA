from openai import OpenAI
from func_timeout import func_set_timeout

client = OpenAI(api_key="")


class GPTGenerator:
    def __init__(self):
        self.retry_times = 3

    def generate(self, prompt, temperature, top_p, model_name):
        out = ''
        for _ in range(self.retry_times):
            # out = self.gpt_chat(prompt, 1)
            try:
                out = self.gpt_chat(prompt, temperature, top_p, model_name)
                break
            except:
                continue
        if out == '':
            return False, out
        else:
            return True, out

    # @func_set_timeout(30)
    # def gpt_completion(self, prompt):
    #     response = openai.Completion.create(
    #         model="gpt-4",
    #         prompt=prompt,
    #         temprature=1,
    #     )
    #     return my_parse(response)

    @func_set_timeout(90)
    def gpt_chat(self, messages, temperature, top_p, model_name):
        messages = [{"role": "system", "content": "You are a helpful assistant."}] + messages
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
        )
        # print(response)
        return my_parse(response)


def my_parse(res):
    if res == '':
        return res
    else:
        return res.choices[0].message.content


if __name__ == '__main__':
    generator = GPTGenerator()
    is_success, output = generator.generate([{"role": "user", "content": "hello"}], 1)
    print(is_success)
    print(output)
