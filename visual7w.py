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
import random


def pil_collate_fn(batch):
    image, question, choices, answer, qa_id, q_type = zip(*batch) 
    return list(image), list(question), list(choices), list(answer), list(qa_id), list(q_type)

class Visual7W(Dataset):
    def __init__(self, image_dir, annotation_file) -> None:
        super().__init__() 

        self.image_dir = image_dir  
        self.datas = [] #42031
        with open(annotation_file, 'r') as file: 
            datas = json.load(file)
        for data in datas['images']:
            if data['split']=='test':
                self.datas.extend(data['qa_pairs'])        
    
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        sample = self.datas[index]
        question = sample['question']
        choices = sample['multiple_choices']
        answer = sample['answer']
        qa_id = sample['qa_id']

        image_id = sample['image_id']        
        image_name = f"v7w_{image_id}.jpg"
        image_path = os.path.join(self.image_dir, image_name)
        image = [Image.open(image_path).convert('RGB')]

        q_type = sample['type']
        return image, question, choices, answer, qa_id, q_type


def parse_args():
    parser = argparse.ArgumentParser(description="VISUAL7W Eval")
    parser.add_argument(
        "--answer_path", type=str, default="results/eval_visual7w.json"
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
    dataset = Visual7W(image_dir=args.image_dir, annotation_file=args.annotation_path) 
    dataloader = DataLoader(dataset, batch_size = args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True, drop_last=False, collate_fn=pil_collate_fn)

    print("Starting...")
    predictions = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, questions, choices, answers, qa_ids, q_types = data
            prompts = []

            for question, choice, answer in zip(questions, choices, answers):
                all_options = choice + [answer]
                random.shuffle(all_options)
                message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {
                                "type": "text",
                                "text": (
                                    "Look at the image carefully and answer this visual question based on the provided choices. "
                                    "Respond with the correct answer only. Do not include any additional text.\n "
                                    f"Question: {question}\n "
                                    f"Choices: \n"
                                    f" {all_options[0]}\n"
                                    f" {all_options[1]}\n"
                                    f" {all_options[2]}\n"
                                    f" {all_options[3]}"
                                )
                            }
                        ]
                    }
                ]
                prompts.append(message)

            for prompt in prompts:
                print(f"Prompt:{prompt}\n", flush=True)

            results = model.generate(prompts=prompts, images=images, max_gen_tokens=25, return_tokens=args.return_tokens)
            # (B, generated_length)
            
            if args.return_tokens:
                for question, pred, qa_id, q_type, answer in zip(questions, results, qa_ids, q_types, answers):
                    predictions.append({
                        'qa_id': qa_id,
                        'answer': pred.tolist(),
                        'gt_answer': answer,
                        'question': question,   
                        'q_type': q_type,            
                    })
            else:
                for question, pred, qa_id, q_type, answer in zip(questions, results, qa_ids, q_types, answers):
                    predictions.append({
                        'qa_id': qa_id,
                        'answer': pred,
                        'gt_answer': answer,
                        'question': question,   
                        'q_type': q_type,              
                    })

    with open(args.answer_path, 'w') as f:
        json.dump(predictions, f, indent=4)

