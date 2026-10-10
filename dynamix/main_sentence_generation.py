import os
import re
import json
import argparse
from utils import generate_sentence, load_configs_new, robusify_Azure_api_call, load_local_chat_api

def append_to_json(file_path, data):
    if os.path.exists(file_path):
        with open(file_path, 'r+') as f:
            existing_data = json.load(f)
            if isinstance(existing_data, list):
                existing_data.append(data)
            else:
                existing_data = [existing_data, data]
            f.seek(0)
            json.dump(existing_data, f, indent=4)
    else:
        with open(file_path, 'w') as f:
            json.dump([data], f, indent=4)

def main(args):
    k = args.num_skills
    index = args.index 
    if args.llm == 'azure':
        robust_Azure_chat_completion = robusify_Azure_api_call(model='gpt-4o-2024-05-13')
    else:
        robust_Azure_chat_completion = load_local_chat_api(base_url=args.llm_base_url, model=args.llm_model, thinking=args.thinking)

    skills, descriptions, categories, refined_config = load_configs_new(robust_Azure_chat_completion, k, config_path=args.config_path, specified_skill=args.specified_skill)
    if args.specified_skill and args.specified_skill not in categories:
        raise ValueError(f"Skill '{args.specified_skill}' not found in the categories: {categories}")
    
    sentence, question = generate_sentence(robust_Azure_chat_completion, skills, descriptions, json_config=refined_config)

    while True:
        if sentence.startswith("WRONG"):
            print("Generating a new sentence...")
            skills, descriptions, categories, refined_config = load_configs_new(robust_Azure_chat_completion, k, config_path=args.config_path, specified_skill=args.specified_skill)
            sentence, question = generate_sentence(robust_Azure_chat_completion, skills, descriptions, json_config=refined_config)
        else:
            print("Correct sentence generated.")
            break
    

    output_dir = args.output_dir if args.specified_skill is None else f"{args.output_dir}/{args.specified_skill}"
    os.makedirs(output_dir, exist_ok=True)

    intermediate_data = {
        'index': index,
        'sentence': sentence,
        'categories': categories,
        'skills': skills,
        'question': question
    }

    intermediate_file_path = os.path.join(output_dir, f'{args.num_iterations}_sentences_k={k}.json')
    append_to_json(intermediate_file_path, intermediate_data)
    
    print(f"Intermediate data appended to {intermediate_file_path}")

    refined_config['index'] = index 
    refined_config_file_path = os.path.join(output_dir, f'{args.num_iterations}_refined_config_k={k}.json')
    append_to_json(refined_config_file_path, refined_config)
    
    print(f"Refined config appended to {refined_config_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images based on provided skills.")
    parser.add_argument("--num_skills", type=int, default=3, help="Number of skills")
    parser.add_argument("--specified_skill", type=str, default=None, help="Specify a skill to generate images for")
    parser.add_argument("--output_dir", type=str, default="all/0_sentence_collection", help="Output directory for generated images")
    parser.add_argument("--config_path", type=str, default="./config", help="Path to the config files")
    parser.add_argument("--index", type=int, default=0, help="Index of the current process")
    parser.add_argument("--num_iterations", type=int, default=300, help="Number of iterations to run")
    parser.add_argument("--llm", type=str, default="local", choices=["local", "azure"], help="Use a local OpenAI-compatible server or Azure GPT-4o")
    parser.add_argument("--llm_base_url", type=str, default=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"), help="Base URL of the local server")
    parser.add_argument("--thinking", action="store_true", help="Enable Qwen3.6 thinking mode (local LLM only)")
    parser.add_argument("--llm_model", type=str, default=os.environ.get("LLM_MODEL", "Qwen3.6-27B"), help="Model name as served by the local server")

    args = parser.parse_args()
    main(args)
