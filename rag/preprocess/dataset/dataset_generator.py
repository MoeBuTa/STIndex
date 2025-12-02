import json
import os
import random
import argparse
import sys

# Add parent directory to path to import ExecutorPrompt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from preprocess.prompts.executor import ExecutorPrompt

def load_and_process_data(data_path, executor_prompt, limit, name):
    data = []

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)

    print(f'Loaded data from {data_path}, total: {len(data)}, limit: {limit}')

    random.shuffle(data)
    data = data[:limit]

    processed_data = []

    # Statistics for supporting_facts length distribution
    supporting_facts_lens = {i: 0 for i in range(1, 11)}

    for item in data:
        question = item['question']
        answers = item['answer_labels']

        # Retrieve context passages
        ctx = item['metadata']['retrieval_contexts']

        # Retrieve supporting facts and match them to context indices
        supporting_facts = item['metadata']['supporting_facts']
        supporting_ids = []
        for supporting_fact in supporting_facts:
            content = supporting_fact['contents']
            support_id = -1
            for i, c in enumerate(ctx):
                if content in c['contents']:
                    support_id = i + 1
                    break
            if support_id == -1:
                print('Error: supporting content not found in contexts')
            supporting_ids.append(support_id)

        supporting_ids = sorted(set(supporting_ids))
        supporting_facts_lens[len(supporting_ids)] += 1

        # Get system and user prompts using executor template
        # Pass ctx directly - it already has 'title' and 'contents' properties
        system_prompt = executor_prompt.get_system_prompt()
        user_prompt = executor_prompt.get_user_prompt(question=question, documents=ctx)

        question_type = item['question_type']

        # Format data in ms-swift GRPO format with messages
        processed_data.append({
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            'answers': answers,
            'question_type': question_type,
            'supporting_ids': supporting_ids,
            'supporting_facts': supporting_facts,
            'name': name,
        })

    print(f'Supporting facts length distribution: {supporting_facts_lens}')
    return processed_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['hotpotqa', 'two_wiki', 'musique'],
                        help='List of dataset names to process')
    parser.add_argument('--train-limits', nargs='+', type=int, default=None,
                        help='Max number of training examples per dataset (must match --datasets length). Use None for all data.')
    parser.add_argument('--test-limit', type=int, default=500,
                        help='Number of test examples per dataset for conversation data')
    parser.add_argument('--grpo-output-name', type=str, default='grpo_train',
                        help='Output filename for merged GRPO training data')
    args = parser.parse_args()

    datasets = args.datasets
    train_limits = args.train_limits
    test_limit = args.test_limit
    grpo_output_name = args.grpo_output_name

    # Validate train_limits
    if train_limits is not None:
        if len(train_limits) != len(datasets):
            raise ValueError(f"--train-limits must have same length as --datasets. Got {len(train_limits)} limits for {len(datasets)} datasets.")
    else:
        # If no limits specified, use None for all datasets (process all data)
        train_limits = [None] * len(datasets)

    random.seed(123456)

    # Initialize executor prompt template
    executor_prompt = ExecutorPrompt()

    # Collect all training data for GRPO (will be merged)
    all_grpo_data = []

    # Process each dataset
    for idx, name in enumerate(datasets):
        train_limit = train_limits[idx]
        print(f'\n{"="*60}')
        print(f'Processing dataset: {name} (train_limit: {train_limit})')
        print(f'{"="*60}')

        # Use absolute paths relative to the script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_root = os.path.join(script_dir, '../../..')

        train_datapath = os.path.join(data_root, f'data/original/{name}/train.jsonl')
        test_datapath = os.path.join(data_root, f'data/original/{name}/dev.jsonl')

        # Process train data (for GRPO)
        train_data = load_and_process_data(train_datapath, executor_prompt, train_limit, name)
        all_grpo_data.extend(train_data)

        # Process test data (for conversation/inference)
        test_data = load_and_process_data(test_datapath, executor_prompt, test_limit, name)

        print(f'Train data size: {len(train_data)}')
        print(f'Test data size: {len(test_data)}')

        # Save conversation data (per dataset, 500 samples each)
        conversation_dir = os.path.join(data_root, f'data/data_conversation/{name}')
        os.makedirs(conversation_dir, exist_ok=True)

        with open(os.path.join(conversation_dir, 'test.jsonl'), 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f'Saved conversation data to: {conversation_dir}/test.jsonl')

    # Shuffle and save merged GRPO training data
    print(f'\n{"="*60}')
    print(f'Creating merged GRPO training data')
    print(f'{"="*60}')

    random.shuffle(all_grpo_data)

    # Add unique IDs to GRPO data
    for i, item in enumerate(all_grpo_data):
        item['id'] = i

    print(f'Total GRPO training samples: {len(all_grpo_data)}')

    # Save GRPO data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(script_dir, '../../..')
    grpo_dir = os.path.join(data_root, 'data/data_train/grpo')
    os.makedirs(grpo_dir, exist_ok=True)

    with open(os.path.join(grpo_dir, f'{grpo_output_name}.jsonl'), 'w', encoding='utf-8') as f:
        for item in all_grpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f'Saved GRPO training data to: {grpo_dir}/{grpo_output_name}.jsonl')

    # Print summary
    print(f'\n{"="*60}')
    print('Summary:')
    print(f'{"="*60}')
    print(f'Datasets processed: {", ".join(datasets)}')
    print(f'Conversation data: {len(datasets)} datasets × {test_limit} samples each')
    print(f'GRPO training data: {len(all_grpo_data)} samples (merged & shuffled)')
    print(f'\nOutput locations:')
    print(f'  - Conversation: data/data_conversation/{{dataset}}/test.jsonl')
    print(f'  - GRPO: data/data_train/grpo/{grpo_output_name}.jsonl')
