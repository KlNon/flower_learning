"""
@Project ：.项目代码 
@File    ：openai
@Describe：
@Author  ：KlNon
@Date    ：2023/3/8 11:50 
"""

import openai

openai.api_key = "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"
prompt = "你是谁?"

response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0.9,
    max_tokens=2048,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.6
)

print(response['choices'][0]['text'])
