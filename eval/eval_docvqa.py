import json
from Levenshtein import distance as levenshtein_distance
import argparse

def NLS(str1, str2, threshold=0.5):
    str1, str2 = str(str1).lower(), str(str2).lower()
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 1.0
    lev_dist = levenshtein_distance(str1, str2)
    norm_lev_dist = lev_dist / max_len
    norm_lev_sim = 1 - norm_lev_dist
    return norm_lev_sim if norm_lev_sim > threshold else 0

def evaluate_ANLS(res_file = "results/eval_docvqa.json"):
    with open(res_file, "r") as f:
        results = json.load(f)

    correct_answers = []
    total_score = 0.0
    for item in results:
        prediction = item["answer"]
        gt_answers = item["gt_answer"]

        max_score = max(NLS(prediction, gt) for gt in gt_answers)
        total_score += max_score
        if max_score > 0:
            item["anls_score"] = round(max_score, 4)  
            correct_answers.append(item)

    for item in correct_answers[:10]:
        print(f"Q: {item['question']}")
        print(f"✓ A: {item['answer']}")
        print(f"GT: {item['gt_answer']}")
        print(f"ANLS Score: {item['anls_score']}")
        print("-" * 50)

    with open("tracing_information_flow/dataset/docvqa/correct_answers.json", "w") as f:
        json.dump(correct_answers, f, indent=2)

    print(f"\n✅ Salvate {len(correct_answers)} risposte corrette su {len(results)} totali.")

    anls_mean = total_score / len(results)
    print(f"\n📊 ANLS medio sul dataset: {round(anls_mean, 4)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DOCVQA results.")
    parser.add_argument("--results", type=str, help="Path to the JSON file with results.", default="results/eval_docvqa.json")
    args = parser.parse_args()
    evaluate_ANLS(args.results)