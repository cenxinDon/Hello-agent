import requests
import json
import os
from tavily import TavilyClient
from openai import OpenAI
import re

AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

#核心规则：
- 必须优先读取用户偏好，推荐时严格匹配（如用户喜欢历史文化，就优先推荐该类型）。
- 若用户提到偏好（如“我喜欢历史景点”“预算200元内”），立即更新用户偏好并记录。
- 若用户拒绝推荐（如“不喜欢这个”），将拒绝计数+1，拒绝3次后必须调整推荐策略（如调整预算）。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str, preferences: str)`: 根据城市、天气、用户偏好搜索推荐的旅游景点（含景点开业状态）。
- `get_alternative(city: str, weather: str, rejected_attractions: list, preferences: str)`: 当景点处于未营业/用户拒绝时，获取备选景点。


# 行动格式:
你的回答必须严格遵循以下格式。首先是你的思考过程，然后是你要执行的具体行动。
Thought: [这里是你的思考过程和下一步计划]
Action: [这里是你要调用的工具，格式为 function_name(arg_name="arg_value")]

# 任务完成:
当你收集到足够的信息，能够回答用户的最终问题时，你必须使用 `finish(answer="...")` 来输出最终答案。

请开始吧！
"""


class OpenAICompatibleClient:
    """
    一个用于调用任何兼容OpenAI接口的LLM服务的客户端。
    """
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """调用LLM API来生成回应。"""
        print("正在调用大语言模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content
            print("大语言模型响应成功。")
            return answer
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return "错误：调用语言模型服务时出错。"


def get_attraction(city: str, weather: str) -> str:
    """
    根据城市和天气，使用Tavily Search API搜索并返回优化后的景点推荐。
    """
    # 1. 从环境变量中读取API密钥
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误：未配置TAVILY_API_KEY环境变量。"

    # 2. 初始化Tavily客户端
    tavily = TavilyClient(api_key=api_key)

    # 3. 构造一个精确的查询
    # query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"

    # 3. 更新后的查询请求
    type_filter = f"，偏好{user_preferences.get('attraction_type', '任意类型')}" if user_preferences.get(
        'attraction_type') else ""
    budget_filter = f"，预算{user_preferences.get('budget', '无限制')}" if user_preferences.get('budget') else ""
    query = f"{city} {weather}天气适合的景点{type_filter}{budget_filter}，需包含门票售罄状态及推荐理由"

    # try:
    #     # 4. 调用API，include_answer=True会返回一个综合性的回答
    #     response = tavily.search(query=query, search_depth="basic", include_answer=True)
    #
    #     # 5. Tavily返回的结果已经非常干净，可以直接使用
    #     # response['answer'] 是一个基于所有搜索结果的总结性回答
    #     if response.get("answer"):
    #         return response["answer"]
    #
    #     # 如果没有综合性回答，则格式化原始结果
    #     formatted_results = []
    #     for result in response.get("results", []):
    #         formatted_results.append(f"- {result['title']}: {result['content']}")
    #
    #     if not formatted_results:
    #         return "抱歉，没有找到相关的旅游景点推荐。"
    #
    #     return "根据搜索，为您找到以下信息：\n" + "\n".join(formatted_results)
    #
    # except Exception as e:
    #     return f"错误：执行Tavily搜索时出现问题 - {e}"

    try:
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
        if response.get("answer"):
            return f"备选景点推荐：{response['answer']}"
        formatted_results = [f"- {r['title']}：{r['content']}" for r in response.get("results", [])]
        return "为您推荐备选景点：\n" + "\n".join(formatted_results) if formatted_results else "暂无合适的备选景点。"
    except Exception as e:
        return f"备选景点查询错误：{e}"


def get_alternative(city: str, weather: str, rejected_attractions: list, preferences: str) -> str:
    """获取备选景点（rejected_attractions为拒绝/售罄的景点列表，json格式）"""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误：未配置TAVILY_API_KEY环境变量。"

    tavily = TavilyClient(api_key=api_key)
    rejected = json.loads(rejected_attractions)
    preferences_dict = json.loads(preferences)
    # 构造查询：排除已拒绝/售罄的景点，同时匹配偏好
    reject_filter = "，排除以下景点：" + "、".join(rejected) if rejected else ""
    query = f"{city} {weather}天气适合的备选景点{reject_filter}，符合{preferences_dict.get('attraction_type', '任意类型')}偏好"

    try:
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
        if response.get("answer"):
            return f"备选景点推荐：{response['answer']}"
        formatted_results = [f"- {r['title']}：{r['content']}" for r in response.get("results", [])]
        return "为您推荐备选景点：\n" + "\n".join(formatted_results) if formatted_results else "暂无合适的备选景点。"
    except Exception as e:
        return f"备选景点查询错误：{e}"

def get_weather(city: str) -> str:
    """
    通过调用 wttr.in API 查询真实的天气信息。
    """
    # API端点，我们请求JSON格式的数据
    url = f"https://wttr.in/{city}?format=j1"

    try:
        # 发起网络请求
        response = requests.get(url)
        # 检查响应状态码是否为200 (成功)
        response.raise_for_status()
        # 解析返回的JSON数据
        data = response.json()

        # 提取当前天气状况
        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_C']

        # 格式化成自然语言返回
        return f"{city}当前天气：{weather_desc}，气温{temp_c}摄氏度"

    except requests.exceptions.RequestException as e:
        # 处理网络错误
        return f"错误：查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        # 处理数据解析错误
        return f"错误：解析天气数据失败，可能是城市名称无效 - {e}"

def update_preferences(current_preferences: dict, update_info: str) -> str:
    """根据用户输入更新偏好（由大模型解析后传入更新信息）"""
    # 示例：大模型会将用户的“我喜欢历史文化，预算300内”解析为结构化更新信息
    # 实际场景中，可让大模型返回json格式的更新内容，这里简化处理
    import json
    try:
        update_data = json.loads(update_info)
        current_preferences.update(update_data)
        # 拒绝计数单独维护（避免大模型误改）
        if "reject_count" in update_data:
            current_preferences["reject_count"] = min(update_data["reject_count"], 3)  # 最多3次
        return f"用户偏好已更新：{json.dumps(current_preferences, ensure_ascii=False)}"
    except Exception as e:
        return f"偏好更新失败：{e}"

# 将所有工具函数放入一个字典，方便后续调用
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
    "update_preferences": update_preferences
}

# --- 1. 配置LLM客户端 ---
# 请根据您使用的服务，将这里替换成对应的凭证和地址
API_KEY = "YOUR_API_KEY"
BASE_URL = "http://192.168.0.112:11434/v1"
MODEL_ID = "deepseek-r1:32b"
TAVILY_API_KEY = "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"
os.environ['TAVILY_API_KEY'] = "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"

llm = OpenAICompatibleClient(
    model=MODEL_ID,
    api_key=API_KEY,
    base_url=BASE_URL
)

# --- 2. 初始化 ---
user_prompt = "你好，请帮我查询一下今天珠海的天气，然后根据天气推荐一个合适的旅游景点。"
prompt_history = [f"用户请求: {user_prompt}"]

user_preferences = {
    "attraction_type":"人造宏伟景象",  # 景点类型（历史文化/自然风景/亲子等）
    "budget": 2000,           # 预算范围（如<100元/100-300元/>300元）
    "avoid": ["爬山""大体力消耗"],              # 避免的类型（如爬山/人多等）
    "reject_count": 0         # 拒绝推荐计数（用于后续策略调整）
}
# 将记忆加入prompt_history，让大模型能读取
prompt_history.append(f"用户当前偏好：{json.dumps(user_preferences, ensure_ascii=False)}")

print(f"用户输入: {user_prompt}\n" + "=" * 40)

# --- 3. 运行主循环 ---
for i in range(5):  # 设置最大循环次数
    print(f"--- 循环 {i + 1} ---\n")

    # 3.1. 构建Prompt
    full_prompt = "\n".join(prompt_history)

    # 3.2. 调用LLM进行思考
    llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
    print(f"模型输出:\n{llm_output}\n")
    prompt_history.append(llm_output)

    # 3.3. 解析并执行行动
    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
    if not action_match:
        print("解析错误：模型输出中未找到 Action。")
        break
    action_str = action_match.group(1).strip()

    if action_str.startswith("finish"):
        final_answer = re.search(r'finish\(answer="(.*)"\)', action_str).group(1)
        print(f"任务完成，最终答案: {final_answer}")
        break

    tool_name = re.search(r"(\w+)\(", action_str).group(1)
    args_str = re.search(r"\((.*)\)", action_str).group(1)
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

    # 新增：处理偏好更新（同步到本地user_preferences）
    if tool_name == "update_preferences":
        update_info = kwargs.get("update_info", "{}")
        observation = available_tools[tool_name](user_preferences, update_info)
        # 同步本地偏好（关键：让后续循环能使用最新偏好）
        user_preferences = json.loads(re.search(r"用户偏好已更新：(.*)", observation).group(1))
    else:
        # 其他工具正常执行
        if tool_name in available_tools:
            observation = available_tools[tool_name](**kwargs)
        else:
            observation = f"错误：未定义的工具 '{tool_name}'"

    # 新增：记录拒绝/售罄的景点（供get_alternative使用）
    rejected_attractions = []
    if "售罄或景点关闭" in observation or "不喜欢" in user_prompt:  # 简化判断，实际可让大模型解析
        # 提取售罄/拒绝的景点名称（需根据observation格式优化正则）
        rejected = re.findall(r"- (.*?)（售罄或景点关闭）", observation)
        rejected_attractions.extend(rejected)
        # 若有售罄，自动触发大模型调用get_alternative
        prompt_history.append(f"Observation: {observation}，已拒绝/售罄关闭景点：{json.dumps(rejected_attractions)}")
    else:
        prompt_history.append(f"Observation: {observation}")

    # 3.4. 记录观察结果
    observation_str = f"Observation: {observation}"
    print(f"{observation_str}\n" + "=" * 40)
    prompt_history.append(observation_str)