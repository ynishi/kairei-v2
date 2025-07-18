import os

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from dotenv import load_dotenv

# .envファイルを読み込む
load_dotenv(".env")

print(f"OPENAI_API_KEY is set: {'OPENAI_API_KEY' in os.environ}")
print(f"ANTHROPIC_API_KEY is set: {'ANTHROPIC_API_KEY' in os.environ}")
print(
    f"Anthropic Key starts with: {os.environ.get('ANTHROPIC_API_KEY', 'NOT SET')[:10]}..."
)

# シンプルなメトリクス
metric = GEval(
    name="Simple Test",
    criteria="Is the output related to the input?",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
    threshold=0.5,
    model="gpt-4o",
)

# テストケース
test_case = LLMTestCase(input="Hello", actual_output="Echo: Hello")

print("Measuring...")
try:
    score = metric.measure(test_case)
    print(f"Score: {metric.score}")
    print(f"Reason: {metric.reason}")
except Exception as e:
    print(f"Error: {e}")
