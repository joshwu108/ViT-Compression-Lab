"""VQA-style accuracy for TextVQA and VQA v2, matching each official evaluator.

EvalAIAnswerProcessor below is copied unchanged from
https://github.com/facebookresearch/mmf/blob/72f898ec35af86567423c7a019dd1dc5175d1a5a/mmf/utils/m4c_evaluators.py
(BSD License, Copyright (c) Facebook, Inc. and its affiliates.)
textvqa_accuracy() reproduces TextVQAAccuracyEvaluator from the same file.
vqav2_accuracy() reproduces VQAEval.evaluate() from the official VQA code:
https://github.com/GT-Vision-Lab/VQA/blob/a013f0043c1e2cdc995922dfe257f7149aa9af06/PythonEvaluationTools/vqaEvaluation/vqaEval.py
Its punctuation, number, article, and contraction rules equal those below.

Usage, from the repository root:

    python scripts/vqa_accuracy.py data/textvqa-pilot/subset.jsonl predictions.json

predictions.json maps record id to the model's answer string. Each record's
"scoring" field selects the function.
"""

import json
import re
import sys
from collections import defaultdict

class EvalAIAnswerProcessor:
    """
    Processes an answer similar to Eval AI
        copied from
        https://github.com/facebookresearch/mmf/blob/c46b3b3391275b4181567db80943473a89ab98ab/pythia/tasks/processors.py#L897
    """

    CONTRACTIONS = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        "Im": "I'm",
        "Ive": "I've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "let's": "let's",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }

    NUMBER_MAP = {
        "none": "0",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    ARTICLES = ["a", "an", "the"]
    PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
    COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
    PUNCTUATIONS = [
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    ]

    def __init__(self, *args, **kwargs):
        pass

    def word_tokenize(self, word):
        word = word.lower()
        word = word.replace(",", "").replace("?", "").replace("'s", " 's")
        return word.strip()

    def process_punctuation(self, in_text):
        out_text = in_text
        for p in self.PUNCTUATIONS:
            if (p + " " in in_text or " " + p in in_text) or (
                re.search(self.COMMA_STRIP, in_text) is not None
            ):
                out_text = out_text.replace(p, "")
            else:
                out_text = out_text.replace(p, " ")
        out_text = self.PERIOD_STRIP.sub("", out_text, re.UNICODE)
        return out_text

    def process_digit_article(self, in_text):
        out_text = []
        temp_text = in_text.lower().split()
        for word in temp_text:
            word = self.NUMBER_MAP.setdefault(word, word)
            if word not in self.ARTICLES:
                out_text.append(word)
            else:
                pass
        for word_id, word in enumerate(out_text):
            if word in self.CONTRACTIONS:
                out_text[word_id] = self.CONTRACTIONS[word]
        out_text = " ".join(out_text)
        return out_text

    def __call__(self, item):
        item = self.word_tokenize(item)
        item = item.replace("\n", " ").replace("\t", " ").strip()
        item = self.process_punctuation(item)
        item = self.process_digit_article(item)
        return item


normalize = EvalAIAnswerProcessor()


def leave_one_out_accuracy(prediction, answers):
    """Mean over leave-one-annotator-out subsets of min(1, matches / 3).

    With 10 answers, a prediction matching 3 or more annotators scores 1.0,
    matching 2 scores 0.6, and matching 1 scores 0.3.
    """
    scores = []
    for left_out in range(len(answers)):
        others = answers[:left_out] + answers[left_out + 1 :]
        scores.append(min(1.0, others.count(prediction) / 3))
    return sum(scores) / len(scores)


def textvqa_accuracy(prediction, human_answers):
    answers = [normalize(a) for a in human_answers]
    return leave_one_out_accuracy(normalize(prediction), answers)


def vqav2_accuracy(prediction, human_answers):
    def strip(text):
        return text.replace("\n", " ").replace("\t", " ").strip()

    def vqa_normalize(text):
        return normalize.process_digit_article(normalize.process_punctuation(text))

    answers = [strip(a) for a in human_answers]
    prediction = strip(prediction)
    # Official quirk: when all annotators agree, neither side is normalized,
    # so "Yes" scores 0 against ten "yes" answers.
    if len(set(answers)) > 1:
        answers = [vqa_normalize(a) for a in answers]
        prediction = vqa_normalize(prediction)
    return leave_one_out_accuracy(prediction, answers)


SCORERS = {
    "textvqa_vqa_accuracy": textvqa_accuracy,
    "vqav2_vqa_accuracy": vqav2_accuracy,
}


def main():
    subset_path, predictions_path = sys.argv[1:3]
    with open(predictions_path) as f:
        predictions = json.load(f)
    with open(subset_path) as f:
        records = [json.loads(line) for line in f]
    by_task = defaultdict(list)
    for r in records:
        if r["id"] in predictions:
            score = SCORERS[r["scoring"]](predictions[r["id"]], r["accepted_answers"])
            by_task[r["task_type"]].append(score)
    scores = [s for task_scores in by_task.values() for s in task_scores]
    print(f"{len(scores)}/{len(records)} predictions scored")
    for task_type, task_scores in sorted(by_task.items()):
        mean = sum(task_scores) / len(task_scores)
        print(f"  {task_type}: {mean:.4f} (n={len(task_scores)})")
    if scores:
        print(f"Overall accuracy: {sum(scores) / len(scores):.4f}")


if __name__ == "__main__":
    main()
