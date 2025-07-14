#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import json
import re
import argparse
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def normalize(word):
    return re.sub(r'[^a-z0-9]', '', word.lower())
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def evaluate_accuracy(results_path):
    correct = 0
    total = 0
    correct_answers = []
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    for entry in data:
        pred = entry["answer"]
        truth = entry["gt_answer"]
        if normalize(pred) == normalize(truth):
            correct += 1
            correct_answers.append(entry)
        else:
            print(f"Predicted: {pred}, Ground Truth: {truth}")
        total += 1

    accuracy = correct / total * 100
    print(f"Correct: {correct}, Total: {total}")
    print(f"Word Accuracy: {accuracy:.2f}%")
    with open("tracing_information_flow/dataset/cocoqa/correct_answers_cocoqa.json", "w") as f:
        json.dump(correct_answers, f, indent=2)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VISUAL7W results.")
    parser.add_argument("--results", type=str, help="Path to the JSON file with results.", default="results/eval_cocoqa.json")
    args = parser.parse_args()
    evaluate_accuracy(args.results)