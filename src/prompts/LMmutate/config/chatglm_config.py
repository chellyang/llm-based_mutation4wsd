import string

import json
from zhipuai import ZhipuAI

client = ZhipuAI(api_key="------------------------------")  # 填写控制台中获取的 APIKey 信息
model = "glm-4"  # 用于配置大模型版本


def getText(role, content, text=None):
    # role 是指定角色，content 是 prompt 内容
    if text is None:
        text = []
    jsoncon = {}
    jsoncon["role"] = role
    jsoncon["content"] = content

    text.append(jsoncon)

    return text


def LLM(promot):
    question = getText("user", promot)
    # 请求模型
    response = client.chat.completions.create(
        model=model,
        messages=question,
        temperature=0.95,
        top_p=0.7,
    )

    content = response.choices[0].message.content
    print(content)
    return content



if __name__ == '__main__':
    sentence = "Now , only one local ringer remains : 64-year-old Derek Hammond ."
    modified_sentence = sentence.replace("``", "")  # 替换引号
