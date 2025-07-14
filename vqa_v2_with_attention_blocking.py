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
    image, question, label, question_id, answer = zip(*batch)  
    return list(image), list(question), list(label), list(question_id), list(answer)

class VQADataset(Dataset):
    def __init__(self, image_dir) -> None:
        super().__init__() 

        self.image_dir = image_dir   
        with open("tracing_information_flow/dataset/vqa/filtered_dataset_vqa.json", "r") as fp:
            self.datas = json.load(fp)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        sample = self.datas[index]
        question_id = sample['question_id']
        answer_tokens = sample['answer_tokens']
        answer = sample['answer']
        image_name = f"COCO_val2014_{sample['image_id']:012d}.jpg"
        image_path = os.path.join(self.image_dir, image_name)
        image = [Image.open(image_path).convert('RGB')]
        question = sample['question']
        return image, question, answer_tokens, question_id, answer


def parse_args():
    parser = argparse.ArgumentParser(description="AVQA Eval")
    parser.add_argument(
        "--answer_path", type=str, default="results/results_cmflowinfo_vqa"
    )
    parser.add_argument(
        "--model_path", type=str, 
    )
    parser.add_argument(
        "--image_dir", type=str, 
    )
    parser.add_argument(
        "--batch_size", type=int, default=1
    )
    parser.add_argument(
        "--start_idx", type=int, default=0
    )
    parser.add_argument(
        "--n_workers", type=int, default=4
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--block_types", nargs='+', type=str, default=['full_attention', 'question_to_last', 'image_to_last', 'image_to_question', 'last_to_last'], choices=['last_to_last', 'image_to_last', 'question_to_last', 'image_to_question', 'full_attention'], help="Blocking types to use"
    )
    parser.add_argument(
        "--k", type=int, default=9, help="Number of blocking window to use"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.answer_path, exist_ok=True) 
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
    dataset = VQADataset(image_dir=args.image_dir) 
    dataloader = DataLoader(dataset, batch_size = args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True, drop_last=False, collate_fn=pil_collate_fn)

    print("Starting...")
    predictions = []
    index = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, questions, answer_tokens, questionIds, answers = data
            prompts = []

            if index < args.start_idx:
                index += len(questions)
                continue

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

            prob_layers = model.generate_multimodal_with_attention_blocking(prompts=prompts, answer_tokens=answer_tokens, images=images, max_gen_len=512, block_types=args.block_types, k=args.k)
            # dict(block_type, list of probabilities) with probabilities = (B, n_layers)/(B)
            sample_dict = dict()

            for idx, question_id in enumerate(questionIds):
                answer_path = args.answer_path + "/" + str(question_id) + ".json"
                for block_type in args.block_types:
                    if block_type in ['last_to_last', 'image_to_last', 'question_to_last', 'image_to_question', 'full_attention']:
                        sample_dict[block_type] = prob_layers[block_type][idx].tolist()
                    else:
                        raise NotImplementedError
                with open(answer_path, 'w') as f:
                    json.dump(sample_dict, f, indent=4)

            index += len(questions)


            