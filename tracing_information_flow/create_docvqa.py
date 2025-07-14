#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import json

with open("tracing_information_flow/dataset/docvqa/correct_answers_docvqa.json", "r") as f:
    results = json.load(f)
    
with open("DocVQA/spdocvqa_qas/val_v1.0_withQT.json", "r") as fp:
    data = json.load(fp)['data']
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correct answer dataset
new_results = []
for result in results:
    if result['anls_score'] < 1.0:
        continue
    question_id = result['questionId']
    samples = [sample for sample in data if sample['questionId']==question_id] 
    assert len(samples)==1
    sample = samples[0]
    # clean from padi_id_tokens
    answer_tokens = result['answer_tokens']
    while answer_tokens[-1] == 128004:
        answer_tokens= answer_tokens[:-1]
        
    result['answer_tokens'] = answer_tokens
    result['question_types'] = sample['question_types']
    result['image'] = sample['image']
    new_results.append(result)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
with open("tracing_information_flow/dataset/chartqa/docvqa_dataset_correct_answers.json", "w") as f:
    json.dump(new_results, f, indent=2)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correct answers ids' file  
id_q_types = {}
for new_result in new_results:
    for q_type in new_result['question_types']:
        if q_type in id_q_types:
            id_q_types[q_type].append(new_result['questionId'])
        else:
            id_q_types[q_type] = [new_result['questionId']]
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
with open("tracing_information_flow/dataset/chartqa/ids_docvqa.json", "w") as f:
    json.dump(id_q_types, f, indent=2)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
question_types_tot = {}
for sample in data:
    for q_type in sample['question_types']:
        if q_type in question_types_tot:
            question_types_tot[q_type].append(sample['questionId'])
        else:
            question_types_tot[q_type] = [sample['questionId']]
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# statistics for each question type
q_types = ['figure/diagram', 'others', 'layout', 'Image/Photo', 'table/list', 'form', 'free_text', 'handwritten', 'Yes/No']
tot_c = 0
for q_type in q_types:
    tot_c += len(id_q_types[q_type])
    print(f"{q_type} correct: {len(id_q_types[q_type])}")
    print(f"{q_type} total: {len(question_types_tot[q_type])}")
    print(f"{q_type} percentage: {len(id_q_types[q_type])/len(question_types_tot[q_type])}")
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# average length of answer tokens for chosen question category 
len_layout_answ = 0
len_table_answ = 0
len_form_answ = 0
len_free_text_answ = 0

len_figure_answ = 0
len_others_answ = 0
len_image_answ = 0
len_handwritten_answ = 0
len_yn_answ = 0

for new_result in new_results:
    if 'layout' in new_result['question_types']:
        len_layout_answ += len(new_result['answer_tokens'])
    if 'table/list' in new_result['question_types']:
        len_table_answ += len(new_result['answer_tokens'])
    if 'form' in new_result['question_types']:
        len_form_answ += len(new_result['answer_tokens'])
    if 'free_text' in new_result['question_types']:
        len_free_text_answ += len(new_result['answer_tokens'])

    if 'figure/diagram' in new_result['question_types']:
        len_figure_answ += len(new_result['answer_tokens'])
    if 'others' in new_result['question_types']:
        len_others_answ += len(new_result['answer_tokens'])
    if 'Image/Photo' in new_result['question_types']:
        len_image_answ += len(new_result['answer_tokens'])
    if 'handwritten' in new_result['question_types']:
        len_handwritten_answ += len(new_result['answer_tokens'])
    if 'Yes/No' in new_result['question_types']:
        len_yn_answ += len(new_result['answer_tokens'])

print(f"layout: {len_layout_answ/len(id_q_types['layout'])}")
print(f"table/list: {len_table_answ/len(id_q_types['table/list'])}")
print(f"form: {len_form_answ/len(id_q_types['form'])}")
print(f"free_text: {len_free_text_answ/len(id_q_types['free_text'])}")

print(f"figure/diagram: {len_figure_answ/len(id_q_types['figure/diagram'])}")
print(f"others: {len_others_answ/len(id_q_types['others'])}")
print(f"Image/Photo: {len_image_answ/len(id_q_types['Image/Photo'])}")
print(f"handwritten: {len_handwritten_answ/len(id_q_types['handwritten'])}")
print(f"Yes/No: {len_yn_answ/len(id_q_types['Yes/No'])}")
