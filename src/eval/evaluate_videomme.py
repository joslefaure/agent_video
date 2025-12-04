import os
import json
import argparse
import logging
from tqdm import tqdm
from src.orchestrator.vllm_agent import VLLMAgent
from src.utils.io import save_json

def evaluate(data_dir: str, anno_dir: str, output_file: str):
    # Load annotations
    # Assuming standard VideoMME structure, usually a json file
    # The user pointed to a folder 'videomme', let's look for json files there
    
    anno_path = os.path.join(anno_dir, "test.json") # Guessing filename
    if not os.path.exists(anno_path):
        # Try to find any json
        files = [f for f in os.listdir(anno_dir) if f.endswith('.json')]
        if files:
            anno_path = os.path.join(anno_dir, files[0])
        else:
            logging.error(f"No annotation file found in {anno_dir}")
            return

    with open(anno_path, 'r') as f:
        dataset = json.load(f)
    
    agent = VLLMAgent(mock=True) # Use mock for now to ensure it runs
    
    results = []
    
    for item in tqdm(dataset):
        video_id = item.get('video_id', '')
        # VideoMME structure might vary, adjusting to common fields
        video_filename = f"{video_id}.mp4"
        video_path = os.path.join(data_dir, video_filename)
        
        if not os.path.exists(video_path):
            # Try searching recursively or checking extensions
            # For now, skip
            continue
            
        question = item['question']
        options = item['options']
        
        try:
            response = agent.run(video_path, question, options)
            
            result_entry = {
                "video_id": video_id,
                "question_id": item.get('question_id'),
                "answer": response['answer_text'],
                "prediction": response
            }
            results.append(result_entry)
        except Exception as e:
            logging.error(f"Error processing {video_id}: {e}")
            
    save_json(results, output_file)
    logging.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--anno_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="results.json")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    evaluate(args.data_dir, args.anno_dir, args.output)
