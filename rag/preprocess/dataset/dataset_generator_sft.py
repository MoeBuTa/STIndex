import json
import os
import random
import argparse
import sys

# Add parent directory to path to import ExecutorPrompt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from preprocess.prompts.executor import ExecutorPrompt


def format_reference_ids(supporting_ids):
    """Format supporting IDs as [1,3,5] format."""
    return '[' + ','.join([str(id) for id in supporting_ids]) + ']'


def load_and_process_data(data_path, executor_prompt, limit, name):
    """Load and process data for SFT."""
    data = []

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)

    print(f'Loaded data from {data_path}, total: {len(data)}, limit: {limit}')

    random.shuffle(data)
    if limit is not None:
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
        system_prompt = executor_prompt.get_system_prompt()
        user_prompt = executor_prompt.get_user_prompt(question=question, documents=ctx)

        # Pick a random answer for SFT training
        answer = random.choice(answers)

        question_type = item['question_type']

        # Format data in ms-swift format
        processed_data.append({
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            'answer': answer,
            'question_type': question_type,
            'supporting_ids': supporting_ids,
            'supporting_facts': supporting_facts,
            'name': name,
        })

    print(f'Supporting facts length distribution: {supporting_facts_lens}')
    return processed_data


def load_inference_results(inference_dir, datasets, project_root):
    """Load filtered inference results from vllm inference."""
    all_data = []

    for name in datasets:
        data_path = os.path.join(project_root, inference_dir, name, 'train.jsonl')
        print(f'\n{"="*60}')
        print(f'Loading filtered inference results from: {name}')
        print(f'{"="*60}')
        print(f'Path: {data_path}')

        if not os.path.exists(data_path):
            print(f'Warning: {data_path} does not exist, skipping...')
            continue

        dataset_count = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                all_data.append(json.loads(line))
                dataset_count += 1

        print(f'Loaded {dataset_count} samples from {name}')

    return all_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['inference', 'direct'], required=True,
                        help='Mode: "inference" for generating prompts, "direct" for complete SFT data')
    parser.add_argument('--datasets', nargs='+', default=['hotpotqa', 'two_wiki', 'musique'],
                        help='List of dataset names to process')
    parser.add_argument('--train-limits', nargs='+', type=int, default=None,
                        help='Max number of training examples per dataset (must match --datasets length)')
    parser.add_argument('--test-limit', type=int, default=500,
                        help='Number of test examples per dataset')
    parser.add_argument('--output-name', type=str, default='sft_25000',
                        help='Output file name for merged SFT training data (used in "direct" mode or when processing inference)')
    parser.add_argument('--inference-dir', type=str, default='results/vllm_inference_results',
                        help='Directory containing vllm inference results (used for processing inference mode results)')
    args = parser.parse_args()

    mode = args.mode
    datasets = args.datasets
    train_limits = args.train_limits
    test_limit = args.test_limit
    output_name = args.output_name
    inference_dir = args.inference_dir

    # Get script directory for paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '../../..')

    random.seed(123456)

    # Initialize executor prompt template
    executor_prompt = ExecutorPrompt()

    # ==================== MODE 1: INFERENCE ====================
    if mode == 'inference':
        print(f'\n{"="*60}')
        print('MODE: INFERENCE - Generating prompts for vllm inference')
        print(f'{"="*60}')

        # Validate train_limits
        if train_limits is not None:
            if len(train_limits) != len(datasets):
                raise ValueError(f"--train-limits must have same length as --datasets. Got {len(train_limits)} limits for {len(datasets)} datasets.")
        else:
            train_limits = [None] * len(datasets)

        # Process each dataset
        for idx, name in enumerate(datasets):
            train_limit = train_limits[idx]
            print(f'\n{"="*60}')
            print(f'Processing dataset: {name} (train_limit: {train_limit})')
            print(f'{"="*60}')

            train_datapath = os.path.join(project_root, f'data/original/{name}/train.jsonl')
            test_datapath = os.path.join(project_root, f'data/original/{name}/dev.jsonl')

            # Process train data (for SFT reasoning generation)
            train_data = load_and_process_data(train_datapath, executor_prompt, train_limit, name)

            # Process test data
            test_data = load_and_process_data(test_datapath, executor_prompt, test_limit, name)

            print(f'Train data size: {len(train_data)}')
            print(f'Test data size: {len(test_data)}')

            # Save conversation data for inference
            conversation_dir = os.path.join(project_root, f'data/data_conversation/{name}')
            os.makedirs(conversation_dir, exist_ok=True)

            with open(os.path.join(conversation_dir, 'train_sft_first_step.jsonl'), 'w', encoding='utf-8') as f:
                for item in train_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            with open(os.path.join(conversation_dir, 'test_sft_first_step.jsonl'), 'w', encoding='utf-8') as f:
                for item in test_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            print(f'Saved SFT first step data to: {conversation_dir}/{{train,test}}_sft_first_step.jsonl')

        # Print summary
        print(f'\n{"="*60}')
        print('Summary:')
        print(f'{"="*60}')
        print(f'Datasets processed: {", ".join(datasets)}')
        print(f'SFT first step data saved for reasoning generation via vllm inference')
        print(f'\nNext steps:')
        print(f'  1. Run vllm inference on the generated data to create reasoning')
        print(f'  2. Filter for correct outputs (EM and source_pick)')
        print(f'  3. Run this script again with mode "inference" to load results and create training pairs')

    # ==================== MODE 2: DIRECT ====================
    elif mode == 'direct':
        print(f'\n{"="*60}')
        print('MODE: DIRECT - Generating complete SFT training data with empty <reasoning>')
        print(f'{"="*60}')

        # For direct mode, use fixed limit of 2000 samples total
        total_samples = 2000
        samples_per_dataset = total_samples // len(datasets)
        remainder = total_samples % len(datasets)

        # Distribute samples evenly, with remainder going to first datasets
        train_limits = []
        for i in range(len(datasets)):
            if i < remainder:
                train_limits.append(samples_per_dataset + 1)
            else:
                train_limits.append(samples_per_dataset)

        print(f'Generating {total_samples} samples total (distribution: {train_limits})')

        all_grpo_data = []

        # Process each dataset
        for idx, name in enumerate(datasets):
            train_limit = train_limits[idx]
            print(f'\n{"="*60}')
            print(f'Processing dataset: {name} (train_limit: {train_limit})')
            print(f'{"="*60}')

            train_datapath = os.path.join(project_root, f'data/original/{name}/train.jsonl')

            # Process train data
            train_data = load_and_process_data(train_datapath, executor_prompt, train_limit, name)
            all_grpo_data.extend(train_data)

            print(f'Train data size: {len(train_data)}')

        # Shuffle and create SFT training data with empty <reasoning>
        print(f'\n{"="*60}')
        print('Creating merged SFT training data with empty <reasoning>')
        print(f'{"="*60}')
        random.shuffle(all_grpo_data)

        dialogs = []
        for i, item in enumerate(all_grpo_data):
            # Format the assistant response with empty <reasoning>
            formatted_response = f"""<reasoning></reasoning>
<sources>{format_reference_ids(item['supporting_ids'])}</sources>
<answer>{item['answer']}</answer>"""

            # Create dialog in ms-swift SFT format
            dialog = {
                "messages": [
                    item['messages'][0],  # system prompt
                    item['messages'][1],  # user prompt
                    {"role": "assistant", "content": formatted_response}
                ],
                "id": i
            }
            dialogs.append(dialog)

        print(f'Total SFT training samples: {len(dialogs)}')

        # Save to output directory
        output_dir = os.path.join(project_root, 'data/data_train/sft')
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f'{output_name}.jsonl')
        with open(output_path, 'w', encoding='utf-8') as f:
            for dialog in dialogs:
                f.write(json.dumps(dialog, ensure_ascii=False) + '\n')

        print(f'Saved SFT training data to: {output_path}')

        # Print summary
        print(f'\n{"="*60}')
        print('Summary:')
        print(f'{"="*60}')
        print(f'Datasets processed: {", ".join(datasets)}')
        print(f'SFT training data: {len(dialogs)} samples (merged & shuffled)')
        print(f'Format: <reasoning></reasoning> (empty), <sources>, <answer>')
        print(f'\nOutput location:')
        print(f'  - SFT: data/data_train/sft/{output_name}.jsonl')
