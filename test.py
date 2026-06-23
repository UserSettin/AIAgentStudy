import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import time

# 加载环境变量
_ = load_dotenv(find_dotenv())

# 配置检查
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")

print(f"API Key: {API_KEY[:10]}...{'*' * 10}" if API_KEY and len(API_KEY) > 10 else "未找到API Key")
print(f"Base URL: {BASE_URL}")
print("-" * 50)

# 初始化OpenAI客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=30.0,  # 增加超时时间
    max_retries=2,  # 重试次数
)


def test_connection():
    """测试连接"""
    try:
        print("正在测试DevAGI连接...")

        # 方法1: 通过models.list测试连接
        print("方法1: 测试模型列表...")
        try:
            models = client.models.list()
            if models and hasattr(models, 'data'):
                print("✓ 模型列表获取成功")
                for model in models.data[:3]:  # 只显示前3个
                    print(f"  - {model.id}")
                return True
        except Exception as e1:
            print(f"⚠ 模型列表获取失败: {e1}")

        # 方法2: 通过简单对话测试
        print("\n方法2: 测试简单对话...")
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "回复'hello'"}],
                max_tokens=5
            )
            if response.choices[0].message.content:
                print(f"✓ 对话测试成功: {response.choices[0].message.content}")
                return True
        except Exception as e2:
            print(f"⚠ 对话测试失败: {e2}")

        return False

    except Exception as e:
        print(f"❌ 连接测试失败: {e}")
        return False


def ask_who_are_you():
    """询问AI身份"""
    try:
        print("\n正在询问AI身份...")
        start_time = time.time()

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个由腾讯开发的AI助手，名字叫元宝。"},
                {"role": "user", "content": "请用中文回答：你是谁？请详细介绍一下自己，包括你的能力、特点、知识范围等。"}
            ],
            temperature=0.7,
            max_tokens=800,
            stream=False
        )

        end_time = time.time()

        # 提取回复
        answer = response.choices[0].message.content
        print("=" * 60)
        print("🤖 AI回答：")
        print("-" * 40)
        print(answer)
        print("-" * 40)

        # 显示统计信息
        usage = response.usage
        print(f"📊 使用统计：")
        print(f"  输入token数：{usage.prompt_tokens}")
        print(f"  输出token数：{usage.completion_tokens}")
        print(f"  总token数：{usage.total_tokens}")
        print(f"  响应时间：{end_time - start_time:.2f}秒")
        print("=" * 60)

        return answer

    except Exception as e:
        print(f"❌ 调用失败：{str(e)}")

        # 更详细的错误信息
        if "401" in str(e):
            print("❌ 认证失败：API Key 无效")
        elif "404" in str(e):
            print("❌ 模型不存在或URL错误")
        elif "timeout" in str(e).lower():
            print("❌ 连接超时，请检查网络")
        elif "connection" in str(e).lower():
            print("❌ 网络连接错误")
        else:
            print("❌ 未知错误")
        return None


if __name__ == "__main__":
    print("🔧 DevAGI OpenAI 连接测试")
    print("=" * 50)

    # 检查环境变量
    if not API_KEY or not BASE_URL:
        print("❌ 请检查.env文件配置是否正确")
        print("   确保包含以下内容：")
        print("   OPENAI_API_KEY=你的API_Key")
        print("   OPENAI_BASE_URL=https://api.fe8.cn/v1")
        exit(1)

    # 测试连接
    if test_connection():
        print("\n✅ 连接成功！开始询问...")
        ask_who_are_you()
    else:
        print("\n❌ 连接测试失败")
        print("\n请按照以下步骤排查：")
        print("1. 📱 确认已注册DevAGI：https://devagi.com")
        print("2. 🔑 获取并复制API Key")
        print("3. 📁 确保.env文件在当前目录")
        print("4. 🌐 检查网络连接（curl https://api.fe8.cn）")
        print("5. 💰 确认账户有可用额度")
        print("6. 🔄 如果还不行，试试其他代理服务（见下方）")