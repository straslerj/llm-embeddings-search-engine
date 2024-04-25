import argparse
import csv
import os
import sys

from tqdm import tqdm

# Add parent directory to path for imports
current_path = os.path.abspath(__file__)
parent_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(parent_path)

import create_vector_store as cvs

LLMS = [
    "gpt2",
    "gpt2-large",
    "distilgpt2",
    "microsoft/phi-2",
    "microsoft/Phi-3-mini-128k-instruct",
]


def get_questions_and_responses(file_path):
    qa_pairs = []

    with open(file_path, "r") as csvfile:
        csvreader = csv.DictReader(csvfile)

        for row in csvreader:
            question = row["question"]
            answer = row["answer"]
            qa_pairs.append((question, answer))

    return qa_pairs


def generate_responses(llms, qa_pairs, mode):
    results = []

    for llm in tqdm(llms, desc="Models"):
        for query, answer in tqdm(qa_pairs, desc="Queries", leave=False):
            doc_chat = cvs.DocChat(model_id=llm, mode=mode)
            response = doc_chat.qNa(query)
            try:
                context, llm_response = response.split("CONTEXT:")[1].split(
                    "ANSWER: ", 1
                )
                context = context.replace("\n", " ").strip()
                llm_response = llm_response.replace("\n", " ").strip()
            except ValueError:
                context = None
                llm_response = None

            results.append((query, answer, llm, context, llm_response))

    directory_path = "responses/"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    with open(f"{directory_path}{mode}_responses.csv", "w") as csvfile:
        fieldnames = ["question", "answer", "llm", "context", "llm_response"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "question": result[0],
                    "answer": result[1],
                    "llm": result[2],
                    "context": result[3],
                    "llm_response": result[4],
                }
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flag for which set of questions and document to use"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["nfl", "syllabus"],
        help="Specify the type (nfl or syllabus)",
    )

    args = parser.parse_args()

    mode_flag = args.mode

    if mode_flag == "nfl":
        print("Creating responses for NFL...")
    elif mode_flag == "syllabus":
        print("Creating responses for syllabus...")
    else:
        print(f"Invalid type: {mode_flag}")
        sys.exit(1)
    questions_file = f"questions/{mode_flag}_questions.csv"
    qa_pairs = get_questions_and_responses(questions_file)
    generate_responses(llms=LLMS, qa_pairs=qa_pairs, mode=mode_flag)
