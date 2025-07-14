import sys
sys.path.append('./')
import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import torch
from PIL import Image
from generation import Llama3_Vision


def pil_collate_fn(batch):
    image, question, label, question_id = zip(*batch) 
    return list(image), list(question), list(label), list(question_id)

class VQADataset(Dataset):
    def __init__(self, image_dir, annotation_file, question_file) -> None:
        super().__init__() 

        self.image_dir = image_dir   
        with open(annotation_file, "r") as fp:
            self.data = json.load(fp)
        with open(question_file, "r") as fp:
            self.question = json.load(fp)
        self.datas = self.data['annotations']
        self.questions = self.question['questions']
    
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        sample = self.datas[index]
        question_id = sample['question_id']
        question = [question['question'] for question in self.questions if question['question_id']==question_id]
        if len(question)>1:
            print(f"More than one question for the question id: {question_id}")
            raise ValueError 
        question = question[0]
        image_name = f"COCO_val2014_{sample['image_id']:012d}.jpg"
        image_path = os.path.join(self.image_dir, image_name)
        image = [Image.open(image_path).convert('RGB')]
        label = sample['answers']
        return image, question, label, question_id


def parse_args():
    parser = argparse.ArgumentParser(description="AVQA Eval")
    parser.add_argument(
        "--answer_path", type=str, default="results/eval_vqa.json"
    )
    parser.add_argument(
        "--model_path", type=str, 
    )
    parser.add_argument(
        "--image_dir", type=str, 
    )
    parser.add_argument(
        "--annotation_path", type=str, 
    )
    parser.add_argument(
        "--question_path", type=str, 
    )
    parser.add_argument(
        "--batch_size", type=int, default=6
    )
    parser.add_argument(
        "--n_workers", type=int, default=4
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument("--return_tokens", action="store_true", help="Ritorna tokens predetti (default: spenta)")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.answer_path), exist_ok=True) 
    for k, v in args._get_kwargs():
        pad = ' '.join(['' for _ in range(25-len(k))])
        print(f"{k}:{pad} {v}", flush=True) 
    torch.cuda.set_device(0)
    torch.manual_seed(1)
    np.random.seed(1)
    
    print('Initializing Model')
    model = Llama3_Vision(args.model_path)
    model.eval()
    print('Initialization Finished')
    dataset = VQADataset(image_dir=args.image_dir,
                         annotation_file=args.annotation_path,
                         question_file=args.question_path) 
    dataloader = DataLoader(dataset, batch_size = args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True, drop_last=False, collate_fn=pil_collate_fn)

    print("Starting...")
    predictions = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, questions, labels, question_ids = data
            prompts = []

            for question in questions:
                message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {
                                "type": "text",
                                "text": (
                                    "Look at the image carefully and answer this visual question. "
                                    "For yes/no questions, just respond Yes or No. "
                                    "If the answer is numeric, just respond with the number and nothing else. "
                                    "If the answer has multiple words, just respond with the words and absolutely nothing else. "
                                    f"Never respond in a sentence or a phrase.\n Respond with as few words as possible.\n Question: {question}"
                                )
                            }
                        ]
                    }
                ]
                prompts.append(message)

            for prompt in prompts:
                print(f"Prompt:{prompt}\n", flush=True)

            results, answer_tokens = model.generate(prompts=prompts, images=images, max_gen_tokens=25, return_tokens=args.return_tokens)
            # (B, generated_length)
            
            for question, pred, answer_token, question_id, label in zip(questions, results, answer_tokens, question_ids, labels):
                predictions.append({
                    'question_id': question_id,
                    'answer': pred,
                    'answer_tokens': answer_token.tolist(),
                    'gt_answer': label,
                    'question': question,                
                })


    with open(args.answer_path, 'w') as f:
        json.dump(predictions, f, indent=4)

