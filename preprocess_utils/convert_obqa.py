import json
import os
import re
import sys
from tqdm import tqdm

__all__ = ['convert_to_obqa_statement']

# String used to indicate a blank
BLANK_STR = "___"


def clean_json_line(json_line):
    def clean_text(text):
        return " ".join(text.strip().split())
    json_line["question"]["stem"] = clean_text(json_line["question"]["stem"])
    for c_idx, choice in enumerate(json_line["question"]["choices"]):
        json_line["question"]["choices"][c_idx]["text"] = clean_text(choice["text"])
    return json_line


def convert_to_obqa_statement(qa_file: str, output_file1: str, clean=False):
    print(f'converting {qa_file} to entailment dataset...')
    os.system('mkdir -p {}'.format(os.path.dirname(output_file1)))
    nrow = sum(1 for _ in open(qa_file, 'r'))
    with open(output_file1, 'w') as output_handle1, open(qa_file, 'r') as qa_handle:
        # print("Writing to {} from {}".format(output_file, qa_file))
        for line in tqdm(qa_handle, total=nrow):
            json_line = json.loads(line)
            if clean:
                json_line = clean_json_line(json_line)
            output_dict = convert_qajson_to_entailment(json_line)
            output_handle1.write(json.dumps(output_dict))
            output_handle1.write("\n")
    print(f'converted statements saved to {output_file1}')
    print()


# Convert the QA file json to output dictionary containing premise and hypothesis
def convert_qajson_to_entailment(qa_json: dict):
    question_text = qa_json["question"]["stem"]
    choices = qa_json["question"]["choices"]
    for choice in choices:
        choice_text = choice["text"]
        statement = question_text + ' ' + choice_text
        create_output_dict(qa_json, statement, choice["label"] == qa_json.get("answerKey", "A"))

    return qa_json


# Create the output json dictionary from the input json, premise and hypothesis statement
def create_output_dict(input_json: dict, statement: str, label: bool) -> dict:
    if "statements" not in input_json:
        input_json["statements"] = []
    input_json["statements"].append({"label": label, "statement": statement})
    return input_json
