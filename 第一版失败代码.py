import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# 加载环境变量
_ = load_dotenv(find_dotenv())

# 初始化OpenAI客户端（使用DevAGI配置）
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)


def ask_who_are_you():
    """询问AI身份"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 或 "gpt-4"
            messages=[
                {"role": "system", "content": "你是一个友好的AI助手。"},
                {"role": "user", "content": "你是谁？请介绍一下自己。"}
            ],
            temperature=0.7,
            max_tokens=500
        )

        # 提取回复
        answer = response.choices[0].message.content
        print("=" * 50)
        print("AI回复：")
        print(answer)
        print("=" * 50)

        # 显示使用情况
        usage = response.usage
        print(f"使用统计：")
        print(f"  输入token数：{usage.prompt_tokens}")
        print(f"  输出token数：{usage.completion_tokens}")
        print(f"  总token数：{usage.total_tokens}")

        return answer

    except Exception as e:
        print(f"调用失败：{e}")
        return None


def test_connection():
    """测试连接和模型列表"""
    try:
        # 测试API连接
        print("正在测试DevAGI连接...")

        # 获取可用模型列表
        models = client.models.list()
        print("可用模型：")
        for model in models.data[:5]:  # 显示前5个模型
            print(f"  - {model.id}")

        return True
    except Exception as e:
        print(f"连接测试失败：{e}")
        return False


if __name__ == "__main__":
    print("DevAGI OpenAI Demo - 你是谁？")
    print("-" * 50)

    # 测试连接
    if test_connection():
        print("连接成功！开始询问...\n")

        # 询问AI身份
        ask_who_are_you()
    else:
        print("请检查：")
        print("1. 是否已注册DevAGI并获取API Key")
        print("2. .env文件配置是否正确")
        print("3. 网络连接是否正常")