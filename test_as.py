import json
import requests

# Set up your OpenAI API key
bearer_token = "ya29.A0AeXRPp74JJu5e7AL9BXkApNo8Vaj4mSSrAxW_-NV0L4EwE1ab19u9qAiJSfL0O10WTpU_6qy5kNZ0k7gQU1vcjjYy3SbZB1wK1pLChN1TW0N-VRNt36cAx4ncnsHawl_oZ1A5LlOPVgYIRc5jdnQrfw624pfY8UeDpcNCiEhF5-ZJT1mr1QZFUwj3hzObCiNlJmhpWmEnGerM7GaV9_ZAxceDuJkZMHC_u7icJnc4U6qvs2PqftKmD_mmEH_170uVVug6ZT8edzqozoP00DJ_SLcc8TL5YpLpWA0qHYZaNR5bTWerBCIsusnhkluIP7Y38VWHgSENmRweyaY0OtMyl-etds_fwtHFMN5prq47M2Oy4IDd7vrhL1SoLJlPRQkXDZlLWHaIMQKVW8cKNg_E5QpGHYwgHXEd1uG-fYR8KZPZcEU_JDDY09MQ-K8SMhHo2QaCgYKAaISARASFQHGX2MiC-Gwv8b6vW8WH78KQDcMyg0458"
auth_project_id = "sparksandbox-genai-sa-argolis"
engine_id = "as-vs-oai_1742947659400"

ENDPOINT = "https://discoveryengine.googleapis.com"
ASSISTANT_NAME = f"projects/{auth_project_id}/locations/global/collections/default_collection/engines/{engine_id}/assistants/default_assistant"
common_limit = "Restrict your answer to 400 words."

from deepeval import evaluate
from deepeval.metrics import GEval, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)

def eval_case(prompt, expected_output, actual_output):
    correctness_metric = GEval(
        name="GEval",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.5,
    )
    test_case = LLMTestCase(
        input=prompt,
        actual_output=actual_output,
        expected_output=expected_output,
    )
    return evaluate([test_case], [correctness_metric, answer_relevancy_metric])


def get_assist_results(query: str):
    response = requests.post(
        f"{ENDPOINT}/v1alpha/{ASSISTANT_NAME}:assist",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {bearer_token}",
            "X-Goog-User-Project": f"{auth_project_id}",
        },
        data=json.dumps({"query": {"text": query + common_limit}}),
    )
    if response.status_code != 200:
        answer = "FAILED"
        assist_token = "None"
    else:
        assist_response = response.json()
        assist_token = assist_response.get("assistToken")
        answer_data = assist_response.get("answer")
        state = answer_data.get("state")
        if state == "SKIPPED":
            answer = answer_data["assistSkippedReasons"][0]
        else:
            answer = (
                answer_data.get("replies")[0]
                .get("groundedContent")
                .get("content")
                .get("text")
            )
    return answer, assist_token


from openai import OpenAI
client = OpenAI()

def call_openai(prompt, model="gpt-3.5-turbo"):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt + common_limit,
            }
        ],
    )
    return completion.choices[0].message.content


def save_response_to_file(response, filename="response.txt"):
    """Saves the API response to a file."""
    with open(filename, "w", encoding="utf-8") as file:
        file.write(response)


def parse_jsonl(file_path):
    """
    Reads a JSONL file and parses each line as a JSON object.

    Args:
        file_path (str): The path to the JSONL file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a parsed JSON object from a line.
              Returns an empty list if the file is empty or encounters errors.
    """
    parsed_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f: #Explicitly use utf-8 encoding
            for line in f:
                try:
                    data = json.loads(line)
                    parsed_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line.strip()}. Error: {e}")
                    # Optionally, you could log the error or continue processing other lines.
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e: #Catch other potential exceptions (IOError, etc)
        print(f"An unexpected error occurred: {e}")

    return parsed_data


if __name__ == "__main__":
    data = parse_jsonl('mt-bench_extended.jsonl')
    for row in data:
        category = row['category']
        user_prompt = row["conversation"][0]['user']
        expected_op = row["conversation"][0]["sys"]
        print(category)
        print(user_prompt)
        response_text_oai = call_openai(user_prompt)
        save_response_to_file(response_text_oai, filename="response_oai.txt")
        assist_results, token = get_assist_results(user_prompt)
        save_response_to_file(assist_results, filename="response_as.txt")
        print(f"Response saved {assist_results}")

        oai_eval = eval_case(
            prompt=user_prompt,
            expected_output=expected_op,
            actual_output=response_text_oai,
        )

        as_eval = eval_case(
            prompt=user_prompt,
            expected_output=expected_op,
            actual_output=assist_results,
        )
        
