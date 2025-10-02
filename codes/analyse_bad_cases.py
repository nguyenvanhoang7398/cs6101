import json
import re
from sklearn.metrics import roc_curve, auc
import json

def calculate_f1_score(correct_true_num, total_true_num, total_false_num, correct_false_num):
    precision_positive = correct_true_num / total_true_num
    recall_true_positive = correct_true_num / (correct_true_num + total_false_num - correct_false_num)
    F1_score_true_positives = 2 * precision_positive * recall_true_positive / (precision_positive + recall_true_positive)

    precision_negative = correct_false_num / total_false_num
    recall_true_negative = correct_false_num / (correct_false_num + total_true_num - correct_true_num)
    F1_score_true_negatives = 2 * precision_negative * recall_true_negative / (precision_negative + recall_true_negative)

    macro_F1_score = (F1_score_true_positives + F1_score_true_negatives) / 2
    return macro_F1_score

def find_common_elements(list_of_sets):
    """
    Finds the common elements (intersection) across a list of sets.

    Args:
        list_of_sets: A list where each element is a set.

    Returns:
        A set containing the elements common to all input sets.
        Returns an empty set if the input list is empty or if there are no common elements.
    """
    if not list_of_sets:
        return set()

    # Initialize the common set with the first set in the list
    common_set = list_of_sets[0].copy()

    # Iterate through the remaining sets and find the intersection
    for current_set in list_of_sets[1:]:
        common_set = common_set.intersection(current_set)
        # Optimization: if common_set becomes empty, no further intersection is needed
        if not common_set:
            break
    return common_set

def get_results_for_invalid_errors_mr_math():
    test_set_filepath = './dataset/mr-math_invalid_errors.json'
    test_dataset = []
    with open(test_set_filepath) as f:
        for line in f:
            test_dataset.append(json.loads(line))

    score_dict = {}
    evaluators = [
                # 'roscoe-sa', 'roscoe-ss', 
                # 'gpt3_5_turbo', 'gpt4', 
                'math-shepherd_mistral-7b', 'reasoneval_llama2-7b',
                'reasoneval_wizardmath-7b-v1.0', 'reasoneval_mistral-7b', 'reasoneval_llemma-7b',
                'reasoneval_abel-7b-002', 'reasoneval_wizardmath-7b-v1.1', 
                # 'reasoneval_llemma-34b'
                ]
    for evaluator in evaluators:
        score_filepath = './eval_results/mr-math_invalid_errors/' + evaluator + '_eval_results.json'
        score_results_record = []
        with open(score_filepath) as f:
            for line in f:
                score_results_record.append(json.loads(line))
        score_dict[evaluator] = score_results_record

    threshold = {'math-shepherd_mistral-7b': 0.5,
                 'reasoneval_llama2-7b': 0.5,
                 'reasoneval_wizardmath-7b-v1.0': 0.5,
                 'reasoneval_mistral-7b': 0.5,
                 'reasoneval_llemma-7b': 0.5,
                 'reasoneval_abel-7b-002': 0.5,
                 'reasoneval_wizardmath-7b-v1.1': 0.5,
                 'reasoneval_llemma-34b': 0.5,
                 'roscoe-sa': 0.025,
                 'roscoe-ss': 0.025}

    ### step-level F1 score

    wrong_false_dict = {evaluator_name: dict() for evaluator_name in evaluators}

    print('**************invalid errors*********step level**********macro f1 score*******')
    for evaluator_name, score_results in score_dict.items():
        correct_true_num = 0
        correct_false_num = 0
        total_false_num = 0
        total_true_num = 0
        if evaluator_name in ['roscoe-sa', 'roscoe-ss']:
            continue
        else:
            for test_data, score_result in zip(test_dataset, score_results):
                example_id = test_data["uuid"] if "uuid" in test_data else test_data["id"]
                gt = []
                if test_data['model_output_solution_correctness'] == 'correct':
                    gt = [1] * len(test_data['model_output_step_format'])
                else:
                    gt = [1] * (int(test_data['model_output_solution_first_error_step']) - 1) + [0] + [
                        'N/A'] * (len(test_data['model_output_step_format']) - int(
                        test_data['model_output_solution_first_error_step']))

                pred = []
                if evaluator_name in ['gpt3_5_turbo', 'gpt4']:
                    if score_result['correctness_pred'] == 'Correct' or score_result['correctness_pred'] == 'correct':
                        pred = [1] * len(test_data['model_output_step_format'])
                    else:
                        step_pred = re.findall('(\d+)', score_result['error_step_pred'])
                        step_pred = step_pred[0] if len(step_pred) > 0 else ''
                        if step_pred != '' and int(step_pred) <= len(test_data['model_output_step_format']):
                            step_pred = int(step_pred)
                            pred = [1] * (step_pred - 1) + [0] + [
                                'N/A'] * (len(test_data['model_output_step_format']) - step_pred)
                        else:
                            ### rare case
                            pred = [1] * len(test_data['model_output_step_format'])
                    assert len(pred) == len(gt)

                else:
                    scores = []
                    if evaluator_name in ['math-shepherd_mistral-7b']:
                        raw_scores = score_result['scores']
                    else:
                        raw_scores = [item[2] + item[1] for item in score_result['scores']]
                    tag_len = 0
                    for step_idx, sub_step in enumerate(test_data['model_output_step_format']):
                        scores.append(min(raw_scores[tag_len:len(sub_step) + tag_len]))
                        tag_len += len(sub_step)
                    assert len(scores) == len(test_data['model_output_step_format'])
                    pred = [1 if score > threshold[evaluator_name] else 0 for score in scores]

                step_index = 1
                error_step = None
                for gt_label, pred_label in zip(gt, pred):
                    if gt_label == 1:
                        total_true_num += 1
                    elif gt_label == 0:
                        total_false_num += 1
                    else:
                        continue
                    if gt_label == pred_label == 0:
                        correct_false_num += 1
                    if gt_label == 0 and pred_label == 1:
                        if example_id not in wrong_false_dict[evaluator_name]:
                            wrong_false_dict[evaluator_name][example_id] = []
                        wrong_false_dict[evaluator_name][example_id].append(step_index)
                        has_error = True
                        if error_step is None:
                            error_step = step_index
                    if gt_label == pred_label == 1:
                        correct_true_num += 1
                    step_index += 1
                if error_step is not None:
                    print("evaluator = {} example_id = {} gt = {} pred = {} first_error_step = {} pred_first_error_step = {}"
                        .format(evaluator_name, example_id, gt, pred, test_data['model_output_solution_first_error_step'], error_step))
        macro_f1_score = calculate_f1_score(correct_true_num, total_true_num, total_false_num, correct_false_num)
        print(f"{evaluator_name}: {format(macro_f1_score, '.3f')}")
        print(f"{evaluator_name}: wrong false case = {format(total_false_num-correct_false_num, '.3f')}")

    # from pprint import pprint
    # pprint(wrong_false_dict)
    all_wrong_false_cases = [set(wrong_false_cases.keys()) for evaluator_name, wrong_false_cases in wrong_false_dict.items() if len(wrong_false_cases) > 0]
    print([len(x) for x in all_wrong_false_cases])
    common_wrong_false_cases = list(find_common_elements(all_wrong_false_cases))
    print("common_wrong_false_cases =", len(common_wrong_false_cases))

    test_data_dict = {row["id"]: row for row in test_dataset}

    wrong_false_out_info = []
    for wrong_case_id in common_wrong_false_cases:
        case_info = test_data_dict[wrong_case_id]
        for evaluator_name in evaluators:
            wrong_evaluator_case_info = wrong_false_dict[evaluator_name][wrong_case_id]
            case_info["{}_wrong_false_step".format(evaluator_name)] = wrong_evaluator_case_info
        wrong_false_out_info.append(case_info)
    
    write_json_lines(wrong_false_out_info, "eval_results/all_evaluator_wrong_false_mr_math_invalid.json")


def get_results_mr_gsm8k():
    test_set_filepath = './dataset/mr-gsm8k.json'
    test_dataset = []
    with open(test_set_filepath) as f:
        for line in f:
            test_dataset.append(json.loads(line))

    evaluator_scores = {}
    evaluator_names = [
        # 'roscoe-sa', 'roscoe-ss', 
        'gpt3_5_turbo', 'gpt4', 'math-shepherd_mistral-7b',
        'reasoneval_llama2-7b', 'reasoneval_wizardmath-7b-v1.0', 'reasoneval_mistral-7b',
        'reasoneval_llemma-7b', 'reasoneval_abel-7b-002', 'reasoneval_wizardmath-7b-v1.1',
        'reasoneval_llemma-34b']
    for name in evaluator_names:
        score_filepath = './eval_results/mr-gsm8k/' + name + '_eval_results.json'
        score_results = []
        with open(score_filepath) as f:
            for line in f:
                score_results.append(json.loads(line))
            evaluator_scores[name] = score_results

    evaluator_thresholds = {'math-shepherd_mistral-7b': 0.5,
                            'reasoneval_llama2-7b': 0.5,
                            'reasoneval_wizardmath-7b-v1.0': 0.5,
                            'reasoneval_mistral-7b': 0.5,
                            'reasoneval_llemma-7b': 0.5,
                            'reasoneval_abel-7b-002': 0.5,
                            'reasoneval_wizardmath-7b-v1.1': 0.5,
                            'reasoneval_llemma-34b': 0.5,
                            'roscoe-sa': 0.025,
                            'roscoe-ss': 0.025}

    ### step-level F1 score

    wrong_false_dict = {evaluator_name: dict() for evaluator_name in evaluator_names}

    print('**************invalid errors*********step level**********macro f1 score*******')
    for evaluator_name, score_results in evaluator_scores.items():
        correct_true_num = 0
        correct_false_num = 0
        total_false_num = 0
        total_true_num = 0
        if evaluator_name in ['roscoe-sa', 'roscoe-ss']:
            continue
        else:
            for i, j in zip(test_dataset, score_results):
                example_id = j["uuid"] if "uuid" in j else j["id"]
                gt = []
                if i['model_output_solution_correctness'] == 'correct':
                    gt = [1] * len(i['model_output_steps'])
                else:
                    gt = [1] * (int(i['model_output_solution_first_error_step']) - 1) + [0] + [
                        'N/A'] * (len(i['model_output_steps']) - int(i['model_output_solution_first_error_step']))

                pred = []
                if evaluator_name in ['gpt3_5_turbo']:
                    if j['gpt3_5_eval_output']['correctness_pred'] == 'Correct' or j['gpt3_5_eval_output'][
                        'correctness_pred'] == 'correct':
                        pred = [1] * len(i['model_output_steps'])
                    else:
                        step_pred = re.findall('(\d+)', j['gpt3_5_eval_output']['error_step_pred'])
                        step_pred = step_pred[0] if len(step_pred) > 0 else ''
                        if step_pred != '' and int(step_pred) <= len(i['model_output_steps']):
                            step_pred = int(step_pred)
                            pred = [1] * (step_pred - 1) + [0] + [
                                'N/A'] * (len(i['model_output_steps']) - step_pred)
                        else:
                            ### rare case
                            pred = [1] * len(i['model_output_steps'])
                    assert len(pred) == len(gt)

                elif evaluator_name in ['gpt4']:
                    if j['gpt4_eval_output']['correctness_pred'] == 'Correct' or j['gpt4_eval_output'][
                        'correctness_pred'] == 'correct':
                        pred = [1] * len(i['model_output_steps'])
                    else:
                        step_pred = re.findall('(\d+)', j['gpt4_eval_output']['error_step_pred'])
                        step_pred = step_pred[0] if len(step_pred) > 0 else ''
                        if step_pred != '' and int(step_pred) <= len(i['model_output_steps']):
                            step_pred = int(step_pred)
                            pred = [1] * (step_pred - 1) + [0] + [
                                'N/A'] * (len(i['model_output_steps']) - step_pred)
                        else:
                            ### rare case
                            pred = [1] * len(i['model_output_steps'])
                    assert len(pred) == len(gt)

                else:
                    scores = []
                    if evaluator_name in ['math-shepherd_mistral-7b']:
                        raw_scores = j['scores']
                    else:
                        raw_scores = [item[2] + item[1] for item in j['scores']]
                    scores = raw_scores
                    assert len(scores) == len(i['model_output_steps'])
                    pred = [1 if score > evaluator_thresholds[evaluator_name] else 0 for score in scores]

                step_index = 1
                for gt_label, pred_label in zip(gt, pred):
                    if gt_label == 1:
                        total_true_num += 1
                    elif gt_label == 0:
                        total_false_num += 1
                    else:
                        continue
                    if gt_label == pred_label == 0:
                        correct_false_num += 1
                    if gt_label == 0 and pred_label == 1:
                        if example_id not in wrong_false_dict[evaluator_name]:
                            wrong_false_dict[evaluator_name][example_id] = []
                        wrong_false_dict[evaluator_name][example_id].append(step_index)
                    if gt_label == pred_label == 1:
                        correct_true_num += 1
                    step_index += 1
        macro_F1_score_step = calculate_f1_score(correct_true_num, total_true_num, total_false_num,
                                                 correct_false_num)
        print(f"{evaluator_name}: {format(macro_F1_score_step, '.3f')}")
        print(f"{evaluator_name}: wrong false case = {format(total_false_num-correct_false_num, '.3f')}")
    
    # from pprint import pprint
    # pprint(wrong_false_dict)
    all_wrong_false_cases = [set(wrong_false_cases.keys()) for evaluator_name, wrong_false_cases in wrong_false_dict.items() if len(wrong_false_cases) > 0]
    print([len(x) for x in all_wrong_false_cases])
    common_wrong_false_cases = list(find_common_elements(all_wrong_false_cases))
    print("common_wrong_false_cases =", len(common_wrong_false_cases))

    test_data_dict = {row["uuid"]: row for row in test_dataset}

    wrong_false_out_info = []
    for wrong_case_id in common_wrong_false_cases:
        case_info = test_data_dict[wrong_case_id]
        for evaluator_name in evaluator_names:
            wrong_evaluator_case_info = wrong_false_dict[evaluator_name][wrong_case_id]
            case_info["{}_wrong_false_step".format(evaluator_name)] = wrong_evaluator_case_info
        wrong_false_out_info.append(case_info)
    
    write_json_lines(wrong_false_out_info, "eval_results/all_evaluator_wrong_false.json")

def write_json_lines(data_list, filename="data_lines.json"):
    """
    Writes a list of Python dictionaries to a file, with each dictionary
    serialized as a single JSON object on a new line.

    This format is commonly known as JSON Lines (or NDJSON).

    Args:
        data_list (list): A list of dictionaries to be written.
        filename (str): The name of the file to write to.
    """
    try:
        # 'w' mode opens the file for writing (and creates it if it doesn't exist)
        with open(filename, 'w', encoding='utf-8') as f:
            print(f"Starting to write {len(data_list)} records to '{filename}'...")

            # Iterate through each dictionary in the list
            for record in data_list:
                # 1. Serialize the Python dictionary 'record' into a JSON string.
                #    ensure_ascii=False allows non-ASCII characters (like ü, é)
                #    to be written directly instead of as \uXXXX escapes.
                json_line = json.dumps(record, ensure_ascii=False)

                # 2. Write the JSON string followed by a newline character (\n).
                f.write(json_line + '\n')

            print(f"Successfully wrote all records to '{filename}'.")

    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # get_results_mr_gsm8k()
    get_results_for_invalid_errors_mr_math()