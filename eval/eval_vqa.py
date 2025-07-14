#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import json
from collections import Counter
import argparse
import string
import re
from word2number import w2n
import contractions
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Handle contractions (e.g., dont → don't)
    text = contractions.fix(text)

    # 3. Convert number words to digits 
    def convert_number_words(match):
        try:
            return str(w2n.word_to_num(match.group(0)))
        except ValueError:
            return match.group(0)  

    text = re.sub(r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|'
                  r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|'
                  r'eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|'
                  r'eighty|ninety|hundred|thousand|million|billion|trillion)(?:[\s-](?:zero|one|two|three|four|five|six|seven|eight|nine|ten|'
                  r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|'
                  r'eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|'
                  r'eighty|ninety))*\b', convert_number_words, text)

    # 4. Remove articles
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 5. Remove periods unless used in decimal numbers
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)  # Remove periods not between digits

    # 6. Replace punctuation with space, except apostrophes and colons
    def replace_punctuation(m):
        char = m.group(0)
        if char == ',':
            before, after = m.start() - 1, m.end()
            if text[before].isdigit() and text[after].isdigit():
                return ','  # Keep comma between digits
            return ''     # Remove comma without space
        return ' '  # Replace other punctuation with space

    text = re.sub(r"[^\w\s':,]", replace_punctuation, text)

    # 7. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def normalize_answer(ans):
    ans = ans.lower().strip()
    # Remove final punctuation if present
    if len(ans) > 0 and ans[-1] in string.punctuation:
        ans = ans[:-1]
    # Remove articles
    ans = re.sub(r'\b(a|an|the)\b', '', ans)
    # Remove punctuation
    ans = re.sub(r'[%s]' % string.punctuation, '', ans)
    # Remove extra whitespace           
    ans = re.sub(r'\s+', ' ', ans)
    return ans
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compute_score(pred_answer, gt_answers):
    pred_answer = preprocess_text(pred_answer)
    gt_answer_list = [preprocess_text(gt["answer"]) for gt in gt_answers]

    # Count matches
    matching = sum([pred_answer == gt for gt in gt_answer_list])

    # Accuracy formula from VQA: min(#matches / 3, 1)
    acc = min(1.0, matching / 3.0)
    perc_matches = float(matching / len(gt_answer_list))
    return acc, perc_matches
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def evaluate_vqa(results_path):
    with open(results_path, 'r') as f:
        data = json.load(f)

    total_score = 0
    num_questions = len(data)
    correct_answers = []
    for entry in data:
        pred_answer = entry["answer"]
        gt_answers = entry["gt_answer"]
        score, perc_matches = compute_score(pred_answer, gt_answers)
        if perc_matches == 1.0:
            #print(f"Correct answer: {pred_answer}")
            correct_answers.append(entry)
        total_score += score

    avg_accuracy = total_score / num_questions
    print(f"Evaluated {num_questions} questions.")
    print(f"Average Accuracy: {avg_accuracy * 100:.2f}%")
    with open("tracing_information_flow/dataset/vqa/correct_answers_vqa.json", "w") as f:
        json.dump(correct_answers, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VQA results.")
    parser.add_argument("--results", type=str, help="Path to the JSON file with VQA results.", default="results/eval_vqa.json")
    args = parser.parse_args()
    evaluate_vqa(args.results)
