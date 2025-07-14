#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import json

with open("tracing_information_flow/dataset/vqa/correct_answers_vqa.json", "r") as f:
    results = json.load(f)

with open("VQA_v2/v2_mscoco_val2014_annotations.json", "r") as fp:
    data = json.load(fp)

with open("VQA_v2/v2_OpenEnded_mscoco_val2014_questions.json", "r") as fp:
    question = json.load(fp)
datas = data['annotations']
questions = question['questions']

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correct answer file
# Pre-processing: creiamo lookup table per accesso rapido
datas_dict = {sample['question_id']: sample for sample in datas}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Costruzione new_results 
new_results = []
for result in results:  # 61350 results
    question_id = result['question_id']
    
    answer_tokens = result['answer_tokens']
    sample = datas_dict[question_id]
    
    # Pulisci dai pad_id_tokens
    while answer_tokens and answer_tokens[-1] == 128004:
        answer_tokens = answer_tokens[:-1]
        
    result['answer_tokens'] = answer_tokens
    
    if '/' in sample['answer_type']:
        answer_type = sample['answer_type'].replace('/', '_')
    else:
        answer_type = sample['answer_type']
    result['answer_type'] = answer_type
    result['image_id'] = sample['image_id']
    
    new_results.append(result)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
with open("tracing_information_flow/dataset/vqa/vqa_dataset_correct_answers.json", "w") as f:
    json.dump(new_results, f, indent=4)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correct answers' ids file  
id_q_types = {}
for result in new_results:
    if '/' in result['answer_type']:
        answer_type = result['answer_type'].replace('/', '_')
    else:
        answer_type = result['answer_type']

    if answer_type in id_q_types:
        id_q_types[answer_type].append(result['question_id'])
    else:
        id_q_types[answer_type] = [result['question_id']]
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
with open("tracing_information_flow/dataset/vqa/ids_vqa.json", "w") as f:
    json.dump(id_q_types, f, indent=2)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
question_types_tot = {}
for sample in datas:
        if '/' in sample['answer_type']:
            answer_type = sample['answer_type'].replace('/', '_')
        else:
            answer_type = sample['answer_type']

        if answer_type in question_types_tot:
            question_types_tot[answer_type].append(sample['question_id'])
        else:
            question_types_tot[answer_type] = [sample['question_id']]
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%           
# statistics for each question type
q_types = ['other', 'yes_no', 'number']
for q_type in q_types:
    print(f"{q_type} correct: {len(id_q_types[q_type])}")
    print(f"{q_type} total: {len(question_types_tot[q_type])}")
    print(f"{q_type} percentage: {len(id_q_types[q_type])/len(question_types_tot[q_type])}")
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# average length of answer tokens for chosen question category 
len_other_answ = 0
len_yn_answ = 0
len_number_answ = 0

for new_result in new_results:
    if 'other' in new_result['answer_type']:
        len_other_answ += len(new_result['answer_tokens'])
    if 'yes_no' in new_result['answer_type']:
        len_yn_answ += len(new_result['answer_tokens'])
    if 'number' in new_result['answer_type']:
        len_number_answ += len(new_result['answer_tokens'])

print(f"other: {len_other_answ/len(id_q_types['other'])}")
print(f"yes_no: {len_yn_answ/len(id_q_types['yes_no'])}")
print(f"number: {len_number_answ/len(id_q_types['number'])}")
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# dataset with 1000 samples for each chosen question category
import random
random.seed(42)

ids_other = [item["question_id"] for item in new_results if 'other' in item['answer_type']]
ids_yn = [item["question_id"] for item in new_results if 'yes_no' in item['answer_type']]
ids_number = [item["question_id"] for item in new_results if 'number' in item['answer_type']]
# %%
dataset={}
list_id =  random.sample(ids_other, 1000)
list_id.sort()
dataset['other'] = list_id
list_id =  random.sample(ids_yn, 1000)
list_id.sort()
dataset['yes_no'] = list_id
list_id =  random.sample(ids_number, 1000)
list_id.sort()
dataset['number'] = list_id
# %%
with open("tracing_information_flow/dataset/vqa/ids_vqa_filtered.json", "w") as file:
    json.dump(dataset, file, indent=4)
# %%
filtered_dataset = []
for result in new_results:
    question_id = result['question_id']
    if question_id in dataset['other'] or question_id in dataset['number'] or question_id in dataset['yes_no']:
        filtered_dataset.append(result)
# %%
with open("tracing_information_flow/dataset/vqa/filtered_dataset_vqa.json", "w") as file:
    json.dump(filtered_dataset, file, indent=4)