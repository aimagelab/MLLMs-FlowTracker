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
    image, question, label, questionId = zip(*batch)  
    return list(image), list(question), list(label), list(questionId)

class DocVQADataset(Dataset):
    def __init__(self, image_dir, annotation_file) -> None:
        super().__init__() 

        self.image_dir = image_dir   
        with open(annotation_file, "r") as fp:
            self.data = json.load(fp)
        self.datas = self.data['data']

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        sample = self.datas[index]
        questionId = sample['questionId']
        label = sample['answers']
        image_name = os.path.basename(sample['image'])
        image_path = os.path.join(self.image_dir, image_name)
        image = [Image.open(image_path).convert('RGB')]
        question = sample['question']
        return image, question, label, questionId


def parse_args():
    parser = argparse.ArgumentParser(description="DOCVQA Eval")
    parser.add_argument(
        "--answer_path", type=str, default="results/eval_docvqa.json"
    )
    parser.add_argument(
        "--annotation_path", type=str, 
    )
    parser.add_argument(
        "--model_path", type=str, 
    )
    parser.add_argument(
        "--image_dir", type=str, 
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
    dataset = DocVQADataset(image_dir=args.image_dir, annotation_file=args.annotation_path) 
    dataloader = DataLoader(dataset, batch_size = args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True, drop_last=False, collate_fn=pil_collate_fn)

    print("Starting...")
    predictions = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, questions, labels, questionIds = data
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
                                    "Read the text in the image carefully and answer the question with the text as seen exactly in the image. "
                                    "For yes/no questions, just respond Yes or No. "
                                    "If the answer is numeric, just respond with the number and nothing else. "
                                    "If the answer has multiple words, just respond with the words and absolutely nothing else. "
                                    f"Never respond in a sentence or a phrase.\n Question: {question}"
                                )
                            }
                        ]
                    }
                ]
                prompts.append(message)

            for prompt in prompts:
                print(f"Prompt:{prompt}\n", flush=True)

            results, answer_tokens = model.generate(prompts=prompts, images=images, max_gen_tokens=512, return_tokens=args.return_tokens)
            # (B, generated_length)
            
            
            for question, pred, answer_token, questionId, label in zip(questions, results, answer_tokens, questionIds, labels):
                predictions.append({
                    'questionId': questionId,
                    'answer': pred,
                    'answer_tokens': answer_token.tolist(), #tensor of generated ids
                    'gt_answer': label,
                    'question': question,                
                })

    with open(args.answer_path, 'w') as f:
        json.dump(predictions, f, indent=4)

