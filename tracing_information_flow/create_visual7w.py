#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import json

with open("tracing_information_flow/dataset/visual7w/correct_answers_visual7w.json", "r") as f:
    results = json.load(f) #35123

datas = [] #42031
with open('Visual7W/dataset_v7w_telling/dataset_v7w_telling.json', 'r') as file: 
    datas_raw = json.load(file)
for data in datas_raw['images']:
    if data['split']=='test':
        datas.extend(data['qa_pairs'])  
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correct answer file
# Pre-processing: creiamo lookup table per accesso rapido
datas_dict = {data['qa_id']: data for data in datas}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correct answer dataset
new_results = [] #1095 samples
for res in results:
    answer_tokens = res['answer_tokens']
    sample = datas_dict[res['qa_id']]

    # clean from padi_id_tokens
    while answer_tokens[-1] == 128004:
        answer_tokens= answer_tokens[:-1]
        
    res['answer_tokens'] = answer_tokens
    res['image_id'] = sample['image_id']
    res['multiple_choices'] = sample['multiple_choices']
    res['type'] = sample['type']

    new_results.append(res)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
with open("tracing_information_flow/dataset/visual7w/visual7w_dataset_correct_answers.json", "w") as f:
    json.dump(new_results, f, indent=2)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correct answers' ids file  
id_q_types = {}
for result in new_results:
    answer_type = result['q_type']

    if answer_type in id_q_types:
        id_q_types[answer_type].append(result['qa_id'])
    else:
        id_q_types[answer_type] = [result['qa_id']]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
with open("tracing_information_flow/dataset/visual7w/ids_visual7w.json", "w") as f:
    json.dump(id_q_types, f, indent=2) 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
question_types_tot = {}
for sample in datas:
        answer_type = sample['type']

        if answer_type in question_types_tot:
            question_types_tot[answer_type].append(sample['qa_id'])
        else:
            question_types_tot[answer_type] = [sample['qa_id']]  
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# statistics for each question type
q_types = ['what', 'where', 'how', 'who', 'when', 'why']
for q_type in q_types:
    print(f"{q_type} correct: {len(id_q_types[q_type])}")
    print(f"{q_type} total: {len(question_types_tot[q_type])}")
    print(f"{q_type} percentage: {len(id_q_types[q_type])/len(question_types_tot[q_type])}")  
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# average length of answer tokens for chosen question category 
len_what_answ = 0
len_where_answ = 0
len_how_answ = 0
len_who_answ = 0
len_when_answ = 0
len_why_answ = 0

for new_result in new_results:
    if 'what' in new_result['q_type']:
        len_what_answ += len(new_result['answer_tokens'])
    if 'where' in new_result['q_type']:
        len_where_answ += len(new_result['answer_tokens'])
    if 'how' in new_result['q_type']:
        len_how_answ += len(new_result['answer_tokens'])
    if 'who' in new_result['q_type']:
        len_who_answ += len(new_result['answer_tokens'])
    if 'when' in new_result['q_type']:
        len_when_answ += len(new_result['answer_tokens'])
    if 'why' in new_result['q_type']:
        len_why_answ += len(new_result['answer_tokens'])

print(f"what: {len_what_answ/len(id_q_types['what'])}")
print(f"where: {len_where_answ/len(id_q_types['where'])}")
print(f"how: {len_how_answ/len(id_q_types['how'])}")
print(f"who: {len_who_answ/len(id_q_types['who'])}")
print(f"when: {len_when_answ/len(id_q_types['when'])}")
print(f"why: {len_why_answ/len(id_q_types['why'])}")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# dataset with 1000 samples for each chosen question category
import random
random.seed(42)

ids_what = [item["qa_id"] for item in new_results if 'what' in item['q_type']]
ids_where = [item["qa_id"] for item in new_results if 'where' in item['q_type']]
ids_how = [item["qa_id"] for item in new_results if 'how' in item['q_type']]
ids_who = [item["qa_id"] for item in new_results if 'who' in item['q_type']]
ids_when = [item["qa_id"] for item in new_results if 'when' in item['q_type']]
ids_why = [item["qa_id"] for item in new_results if 'why' in item['q_type']]

# %%
dataset={}
list_id =  random.sample(ids_what, 1000)
list_id.sort()
dataset['what'] = list_id
list_id =  random.sample(ids_where, 1000)
list_id.sort()
dataset['where'] = list_id
list_id =  random.sample(ids_how, 1000)
list_id.sort()
dataset['how'] = list_id
list_id =  random.sample(ids_who, 1000)
list_id.sort()
dataset['who'] = list_id
list_id =  random.sample(ids_when, 1000)
list_id.sort()
dataset['when'] = list_id
list_id =  random.sample(ids_why, 1000)
list_id.sort()
dataset['why'] = list_id
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
with open("tracing_information_flow/dataset/visual7w/ids_visual7w_filtered.json", "w") as file:
    json.dump(dataset, file, indent=4)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filtered_dataset = []
for result in new_results:
    qa_id = result['qa_id']
    if qa_id in dataset['what'] or qa_id in dataset['where'] or qa_id in dataset['how'] or qa_id in dataset['when'] or qa_id in dataset['why'] or qa_id in dataset['who']:
        filtered_dataset.append(result)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
with open("tracing_information_flow/dataset/visual7w/filtered_dataset_visual7w.json", "w") as file:
    json.dump(filtered_dataset, file, indent=4)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
