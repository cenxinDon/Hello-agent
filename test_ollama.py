from openai import OpenAI

client = OpenAI(
    api_key="test",  # 任意非空字符串
    base_url="http://192.168.0.112:11434/v1"  # 替换为你的地址
)

try:
    # 列出可用模型
    models = client.models.list()
    print("可用模型：")
    for model in models.data:
        print(f"- {model.id}")

    # 简单对话测试
    response = client.chat.completions.create(
        model=models.data[0].id,  # 使用第一个可用模型
        messages=[{"role": "user", "content": "hello"}]
    )
    print("\n模型回复：", response.choices[0].message.content)
except Exception as e:
    print("连接错误：", e)
