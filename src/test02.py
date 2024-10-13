def GenPrompt(from_lang, to_lang, text):
    return """
    你是一个专业的翻译助手。请将下面<data>标签中的{}翻译成{}，你只需要回答翻译结果。     
    <data>{}</data>
    """.format(
        from_lang, to_lang, text
    )


print(GenPrompt(from_lang=1, to_lang=2, text=3))

# promptTemplate的基本使用

from langchain.prompts.prompt import PromptTemplate

promptTemplate = PromptTemplate.from_template("简单介绍下{city}这座城市的特色")
print(promptTemplate.format(city="北京"))


# prompt提供少量示例

examples = [
    {
        "question": "从一副标准扑克牌中随机抽取一张牌，计算抽到红桃的概率。",
        "answer": "概率为 1/4。",
    },
    {
        "question": "在一组人中，60%是女性。如果随机选择一个人，计算她是女性并且是左撇子的概率，已知左撇子的概率为 10%。",
        "answer": "概率为 6%。",
    },
    {
        "question": "一枚硬币被抛两次，计算至少一次出现正面的概率。",
        "answer": "概率为 3/4。",
    },
    {
        "question": "在一批产品中，90%是正常的，10%有缺陷。如果一个产品有缺陷，被检测到的概率是95%。计算一个被检测为有缺陷的产品实际上是有缺陷的概率。",
        "answer": "概率为 32/143。",
    },
    {
        "question": "一个骰子被掷 3 次，计算得到至少一次 6 的概率。",
        "answer": "概率为 91/216。",
    },
    {
        "question": "在 0 到 1 的区间内均匀分布的随机变量 X，计算 X 小于 0.3 的概率。",
        "answer": "概率为 0.3。",
    },
    {
        "question": "一枚硬币被抛 5 次，计算正面朝上的次数为 3 的概率。",
        "answer": "概率为 10/32。",
    },
    {
        "question": "考试成绩近似服从正态分布，平均分为 70 分，标准差为 10 分。计算得分高于 80 分的学生的概率。",
        "answer": "概率为约 15.87%。",
    },
]

example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Question: {question}\n{answer}"
)

from langchain.prompts.few_shot import FewShotPromptTemplate

few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="You are an assistant good at {field}, learn the below examples and then answer the last question",
    suffix="Question: {question}",
    input_variables=["field", "question"],
)

# print(few_shot_prompt_template.format(field="math", question="一个标准六面骰子被投掷一次，计算出现奇数的概率。"))

# 动态样本提示，由于token是有限的，所以每次发全部example不现实
from langchain.prompts.example_selector import LengthBasedExampleSelector


def get_len_by_char(text: str):
    return len(text)


example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=100,
    get_text_length=get_len_by_char,
)

few_shot_prompt_template2 = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="You are an assistant good at {field}, learn the below examples and then answer the last question",
    suffix="Question: {question}",
    input_variables=["field", "question"],
)

print(
    few_shot_prompt_template2.format(
        field="math", question="一个标准六面骰子被投掷一次，计算出现奇数的概率。"
    )
)
