import json
import argparse
import os

def extract_sample_tokens(json_path):
    with open(json_path) as f:
        qa_data = json.load(f)
    tokens = sorted(set(q['sample_token'] for q in qa_data['questions']))
    return tokens

def main():
    parser = argparse.ArgumentParser(description="Extract sample_tokens from NuScenes-QA JSON.")
    parser.add_argument('--json-path', type=str, required=True, help='Path to NuScenes-QA question JSON file')
    parser.add_argument('--out-path', type=str, default=None, help='Path to save output txt file')

    args = parser.parse_args()

    tokens = extract_sample_tokens(args.json_path)
    
    # Default output path if not provided
    if args.out_path is None:
        base_name = os.path.splitext(os.path.basename(args.json_path))[0]
        args.out_path = f'data/nuscenes/qa_tokens_{base_name.replace("NuScenes_", "").replace("_questions", "")}.txt'

    with open(args.out_path, 'w') as f:
        f.write('\n'.join(tokens))

    print(f"Saved {len(tokens)} sample tokens to {args.out_path}")

if __name__ == '__main__':
    main()
