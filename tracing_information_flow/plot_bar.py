#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import sys
sys.path.append('./')
import ast
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
datas = [
    json.load(open('tracing_information_flow/dataset/docvqa/docvqa_dataset_correct_answers.json')),
    json.load(open('tracing_information_flow/dataset/vqa/filtered_dataset_vqa.json')),
    json.load(open('tracing_information_flow/dataset/visual7w/filtered_dataset_visual7w.json'))
]
list_ids = [
    json.load(open('tracing_information_flow/dataset/docvqa/ids_docvqa.json')),
    json.load(open('tracing_information_flow/dataset/vqa/ids_vqa_filtered.json')),
    json.load(open(f"tracing_information_flow/dataset/visual7w/ids_visual7w_filtered.json"))
]
    
datasets = [
    'DOCVQA',
    'VQA', 
    'visual7w']

answer_paths = [
    "tracingresults/results_cmflowinfo_docvqa",
    "tracingresults/results_cmflowinfo_vqa_k_9",
    "tracingresults/results_cmflowinfo_visual7w"
]

stages = {
    "1-3": range(0, 2),
    "4-8": range(3, 7),
    "9-13": range(8, 12),
    "14-18": range(13, 17),
    "19-23": range(18, 22),
    "24-28": range(23, 27),
    "29-33": range(28, 32),
    "34-38": range(33, 37),
    "39-40": range(38, 40),
}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
### Read result file
eps=1e-5
layers_change_probs_datasets = {}
for i, d in enumerate(datas):
    print(f"Dataset: {datasets[i]}")
    answer_path = answer_paths[i]
    count = 0
    list_id = list_ids[i]
    layers_change_probs = {}
    for category, ids in list_id.items():
        for id in ids: #for each sample in a category
            path = answer_path + "/" + str(id) + ".json"
            if os.path.exists(path):
                sample_probs = json.load(open(path))
                p1 = sample_probs['full_attention'] + eps
                if p1 < 0.5:
                    continue
                count += 1
                ### Experiment 1
                if 'last_to_last' in sample_probs:
                    if 'last_to_last' in layers_change_probs:
                        change_prob = change_prob = [((p2 - p1) / p1) * 100 for p2 in sample_probs['last_to_last']]
                        layers_change_probs['last_to_last'] = [(a + b) for a, b in zip(layers_change_probs['last_to_last'], change_prob )]
                    else:
                        layers_change_probs['last_to_last'] = [((p2 - p1) / p1) * 100 for p2 in sample_probs['last_to_last']]

                if 'question_to_last' in sample_probs:
                    if 'question_to_last' in layers_change_probs:
                        change_prob = [((p2 - p1) / p1) * 100  for p2 in sample_probs['question_to_last']]
                        layers_change_probs['question_to_last'] = [(a + b) for a, b in zip(layers_change_probs['question_to_last'], change_prob )]
                    else:
                        layers_change_probs['question_to_last'] =  [((p2 - p1) / p1) * 100  for p2 in sample_probs['question_to_last']]
                
                if 'image_to_last' in sample_probs:
                    if 'image_to_last' in layers_change_probs:
                        change_prob = [((p2 - p1) / p1) * 100  for p2 in sample_probs['image_to_last']]
                        layers_change_probs['image_to_last'] = [(a + b) for a, b in zip(layers_change_probs['image_to_last'], change_prob )]
                    else:
                        layers_change_probs['image_to_last'] = [((p2 - p1) / p1) * 100  for p2 in sample_probs['image_to_last']]
                
                ### Experiment 2
                if 'image_to_question' in sample_probs:
                    if 'image_to_question' in layers_change_probs:
                        change_prob = [((p2 - p1) / p1) * 100 for p2 in sample_probs['image_to_question']]
                        layers_change_probs['image_to_question'] = [(a + b) for a, b in zip(layers_change_probs['image_to_question'], change_prob )]
                    else:
                        layers_change_probs['image_to_question'] = [((p2 - p1) / p1) * 100  for p2 in sample_probs['image_to_question']]

            else:
                continue
        
    for key in layers_change_probs.keys():
        layers_change_probs[key] = [x / count for x in layers_change_probs[key]]
    
    print(f"count: {count}")
    layers_change_probs_datasets[datasets[i]] = layers_change_probs
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# PLOT: LinePLot dei flussi I->Q, Q->L, I->L
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

docvqa_i2q = layers_change_probs_datasets['DOCVQA']['image_to_question']
vqa_open_i2q = layers_change_probs_datasets['VQA']['image_to_question']
vqa_mc_i2q = layers_change_probs_datasets['visual7w']['image_to_question']

docvqa_q2l = layers_change_probs_datasets['DOCVQA']['question_to_last']
vqa_open_q2l = layers_change_probs_datasets['VQA']['question_to_last']
vqa_mc_q2l = layers_change_probs_datasets['visual7w']['question_to_last']

docvqa_i2l = layers_change_probs_datasets['DOCVQA']['image_to_last']
vqa_open_i2l = layers_change_probs_datasets['VQA']['image_to_last']
vqa_mc_i2l = layers_change_probs_datasets['visual7w']['image_to_last']

labels = list(stages.keys())

def stage_means(data):
    return [abs(np.mean([data[i] for i in idxs])) for idxs in stages.values()]

docvqa_i2q_means = stage_means(docvqa_i2q)
vqa_open_i2q_means = stage_means(vqa_open_i2q)
vqa_mc_i2q_means = stage_means(vqa_mc_i2q)

docvqa_i2l_means = stage_means(docvqa_i2l)
vqa_open_i2l_means = stage_means(vqa_open_i2l)
vqa_mc_i2l_means = stage_means(vqa_mc_i2l)

docvqa_q2l_means = stage_means(docvqa_q2l)
vqa_open_q2l_means = stage_means(vqa_open_q2l)
vqa_mc_q2l_means = stage_means(vqa_mc_q2l)

data_i2q = {
    'Visual7W': vqa_mc_i2q_means,
    'VQAv2': vqa_open_i2q_means,
    'DocVQA': docvqa_i2q_means,
}
data_i2l = {
    'Visual7W': vqa_mc_i2l_means,
    'VQAv2': vqa_open_i2l_means,
    'DocVQA': docvqa_i2l_means,
}
data_q2l = {
    'Visual7W': vqa_mc_q2l_means,
    'VQAv2': vqa_open_q2l_means,
    'DocVQA': docvqa_q2l_means,
}
df_i2q = pd.DataFrame(data_i2q, index=labels).T  # Rows = datasets, Columns = stages
df_i2l = pd.DataFrame(data_i2l, index=labels).T
df_q2l = pd.DataFrame(data_q2l, index=labels).T


fig, axes = plt.subplots(1, 3, figsize=(22, 6))

# Plot i2q
df_i2q.T.plot(ax=axes[0], marker='o', grid=True)
axes[0].set_title('Image ↛ Question', fontsize=30)
axes[0].set_xlabel('Layers', fontsize=20)
axes[0].set_ylabel('Change in probability (%)', fontsize=20)
axes[0].set_ylim(0, 105)

# Plot i2l
df_i2l.T.plot(ax=axes[1], marker='o', grid=True)
axes[1].set_title('Image ↛ Last', fontsize=30)
axes[1].set_xlabel('Layers', fontsize=20)
#axes[1].set_ylabel('Change in probability (%)') 
axes[1].set_ylim(0, 105)

# Plot q2l
df_q2l.T.plot(ax=axes[2], marker='o', grid=True)
axes[2].set_title('Question ↛ Last', fontsize=30)
axes[2].set_xlabel('Layers', fontsize=20)
#axes[2].set_ylabel('Change in probability (%)')  
axes[2].set_ylim(0, 105)

handles, labels = axes[0].get_legend_handles_labels()
for ax in axes:
    ax.legend_.remove()
fig.legend(handles, labels, loc='lower right', ncol=3, fontsize=30, bbox_to_anchor=(0.90, -0.1), frameon=True)
fig.subplots_adjust(bottom=0.25, left=0.1)
fig.tight_layout(rect=[0, 0.05, 1, 1])  
plt.savefig(f'tracing_information_flow/visualization/line_plot_grid.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.show()
# %%
