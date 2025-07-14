#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import sys
sys.path.append('./')
import ast
import os
import json
import matplotlib.pyplot as plt
import numpy as np
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#PLOT DOCVQA
datas = json.load(open('tracing_information_flow/dataset/docvqa/docvqa_dataset_correct_answers.json'))
list_ids = json.load(open('tracing_information_flow/dataset/docvqa/ids_docvqa.json'))

map_dict = {
    "last_to_last": "Last ↛ Last",
    "image_to_last": "Image ↛ Last",
    "question_to_last": "Question ↛ Last",
    "image_to_question": "Image ↛ Questiont",
}
x_min = 0
x_max = 40
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
### Experiment 1+2: Modality_to_Last + Modality_to_Modality (last_to_last, image_to_last, question_to_last, image_to_question)
answer_path = "results/results_cmflowinfo_docvqa"
eps=1e-5
count_tot = 0
for category, ids in list_ids.items():
    count = 0
    if '/' in category:
        category = category.replace('/', '_')
    layers_change_probs = {}
    for id in ids: #for each sample in a category
        path = answer_path + "/" + str(id) + ".json"
        if os.path.exists(path):
            sample_probs = json.load(open(path))
            p1 = sample_probs['full_attention'] + eps
            #sample = [s for s in datas if s['questionId'] == id][0]
            #if len(sample['answer_tokens']) > 8 or p1 < 0.5:
            if p1 < 0.5:
                continue
            count += 1
            ### Experiment 1
            if 'last_to_last' in sample_probs:
                if 'last_to_last' in layers_change_probs:
                    change_prob = change_prob = [((p2 - p1) / p1) * 100 if ((p2 - p1) / p1)<0 else 0 for p2 in sample_probs['last_to_last']]

                    layers_change_probs['last_to_last'] = [(a + b) for a, b in zip(layers_change_probs['last_to_last'], change_prob )]
                else:
                    layers_change_probs['last_to_last'] = [((p2 - p1) / p1) * 100 if ((p2 - p1) / p1)<0 else 0 for p2 in sample_probs['last_to_last']]

            if 'question_to_last' in sample_probs:
                if 'question_to_last' in layers_change_probs:
                    change_prob = [((p2 - p1) / p1) * 100  for p2 in sample_probs['question_to_last']]
                    layers_change_probs['question_to_last'] = [(a + b) for a, b in zip(layers_change_probs['question_to_last'], change_prob )]
                else:
                    layers_change_probs['question_to_last'] =  [((p2 - p1) / p1) * 100  for p2 in sample_probs['question_to_last']]
            
            if 'image_to_last' in sample_probs:
                if 'image_to_last' in layers_change_probs:
                    change_prob = [((p2 - p1) / p1) * 100 if ((p2 - p1) / p1)<0 else 0 for p2 in sample_probs['image_to_last']]
                    layers_change_probs['image_to_last'] = [(a + b) for a, b in zip(layers_change_probs['image_to_last'], change_prob )]
                else:
                    layers_change_probs['image_to_last'] = [((p2 - p1) / p1) * 100 if ((p2 - p1) / p1)<0 else 0 for p2 in sample_probs['image_to_last']]
            
            ### Experiment 2
            if 'image_to_question' in sample_probs:
                if 'image_to_question' in layers_change_probs:
                    change_prob = [((p2 - p1) / p1) * 100 if ((p2 - p1) / p1)<0 else 0 for p2 in sample_probs['image_to_question']]
                    layers_change_probs['image_to_question'] = [(a + b) for a, b in zip(layers_change_probs['image_to_question'], change_prob )]
                else:
                    layers_change_probs['image_to_question'] = [((p2 - p1) / p1) * 100 if ((p2 - p1) / p1)<0 else 0 for p2 in sample_probs['image_to_question']]

        else:
            #print(f"file json {path} does not exist!!")
            continue
    
    for key in layers_change_probs.keys():
        layers_change_probs[key] = [x / count for x in layers_change_probs[key]]
    
    count_tot += count
    print(f"{category}->count: {count}")

    # Experiment 1
    if 'last_to_last' in layers_change_probs and 'question_to_last' in layers_change_probs and 'image_to_last' in layers_change_probs and 'image_to_question' in layers_change_probs and 'image_to_question' in layers_change_probs:
        layers = np.arange(len(layers_change_probs['last_to_last']))
        # === First Plot: Information Flow Curves ===
        plt.figure(figsize=(4, 3)) 
        plt.plot(layers, layers_change_probs['last_to_last'], color='red', linewidth=2, alpha=0.9)
        plt.plot(layers, layers_change_probs['image_to_last'], color='orange', linewidth=2, alpha=0.9)
        plt.plot(layers, layers_change_probs['question_to_last'], color='green', linewidth=2, alpha=0.9)
        plt.plot(layers, layers_change_probs['image_to_question'], color='purple', linewidth=2, alpha=0.9)
        plt.xlabel("Layer", fontsize=12)
        plt.ylabel("Change in probability (%)", fontsize=12)
        plt.axvline(x=3, color='blue', linestyle=':')
        plt.axvline(x=8, color='blue', linestyle=':')
        plt.axvline(x=13, color='blue', linestyle=':')
        plt.axvline(x=18, color='blue', linestyle=':')
        plt.axvline(x=23, color='blue', linestyle=':')
        plt.axvline(x=28, color='blue', linestyle=':')
        plt.axvline(x=33, color='blue', linestyle=':')
        plt.axvline(x=38, color='blue', linestyle=':')
        plt.grid(True)
        #plt.show()
        plt.savefig(f'tracing_information_flow/figures/docvqa/grafico_{category}.pdf', format='pdf', dpi=600, bbox_inches='tight')
print(f"count tot: {count_tot}")