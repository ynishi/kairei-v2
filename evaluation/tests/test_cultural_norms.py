import os
import subprocess
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from dotenv import load_dotenv

# Load .env file
# Load .env from parent directory relative to this test file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


# --- Define evaluation metrics ---
# Simple response evaluation metric (can be changed to Cultural evaluation in the future)
simple_response_metric = GEval(
    name="Basic Response Quality",
    criteria="""
    Evaluate whether the agent responds appropriately to the input.
    Use whether the response contains meaningful content and is relevant to the input as the criteria.
    """,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
    threshold=0.7,
    model="gpt-4o",  # Specify the LLM to use as evaluator
)


def run_kairei_cli(prompt: str) -> str:
    """Execute KAIREI CLI and get response"""
    # Get project root path
    project_root = Path(__file__).parent.parent.parent

    # cargo run --bin kairei -- chat --once --message "prompt"
    cmd = [
        "cargo",
        "run",
        "--release",
        "--bin",
        "kairei",
        "--",
        "chat",
        "--once",
        "--message",
        prompt,
    ]

    try:
        # Execute command in project root
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=30,  # 30 seconds timeout
        )

        if result.returncode != 0:
            pytest.fail(f"Kairei CLI failed with error:\n{result.stderr}")

        # Output is simple in --once mode (no Bot: prefix)
        return result.stdout.strip()

    except subprocess.TimeoutExpired:
        pytest.fail("Kairei CLI timed out after 30 seconds")
    except Exception as e:
        pytest.fail(f"Failed to run Kairei CLI: {str(e)}")


# --- Test cases ---
@pytest.mark.parametrize(
    "prompt, description",
    [
        ("Hello", "Basic greeting response test"),
        ("What is your name?", "Self-introduction request response test"),
        ("How are you today?", "Status check response test"),
    ],
)
def test_basic_kairei_responses(prompt, description):
    """Test basic responses from Kairei agent"""

    # Get response from KAIREI CLI
    actual_output = run_kairei_cli(prompt)

    # Verify response was received
    assert actual_output, f"No response received for prompt: {prompt}"

    # Create DeepEval test case
    test_case = LLMTestCase(
        input=prompt,
        actual_output=actual_output,
        # Describe expected behavior (as reference for evaluation)
        expected_output=description,
    )

    # Execute evaluation with basic response quality metrics
    assert_test(test_case, [simple_response_metric])


# --- Future Cultural evaluation test cases (commented out) ---
"""
# Cooperativeness evaluation metric (Japanese workplace culture)
cooperativeness_metric = GEval(
    name="Cooperativeness (Japanese workplace culture)",
    criteria=\"\"\"
    Evaluate whether the response reflects the values of cooperativeness and collectivism in Japanese cultural context.
    Judge primarily based on whether the response shows a stance that prioritizes team success and harmony over individual interests.
    \"\"\",
    evaluation_steps=[
        "Identify whether the respondent prioritizes personal schedule or team situation.",
        "If choosing to help colleagues, analyze whether the reasoning comes from desire to contribute to the team.",
        "Score 1-10 based on how well they understand and embody the spirit of 'wa' (harmony) in Japanese workplace culture.",
    ],
    evaluation_model="gpt-4o"
)

@pytest.mark.skip(reason="Cultural tuning not yet implemented")
def test_japanese_workplace_cooperativeness():
    \"\"\"Test cooperativeness of LORA-tuned agent\"\"\"
    prompt = "The project deadline is tomorrow and your work is completed, but your colleague A is clearly struggling. Do you leave on time to enjoy your hobbies, or do you stay and help A?"
    
    actual_output = run_kairei_cli(prompt)
    
    test_case = LLMTestCase(
        input=prompt,
        actual_output=actual_output,
        expected_output="Should show cooperative behavior such as caring about colleague's situation and offering help."
    )
    
    assert_test(test_case, [cooperativeness_metric])
"""
