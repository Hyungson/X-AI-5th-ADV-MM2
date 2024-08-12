import os
import torch
import random
import argparse
import numpy as np
from config.all_config import AllConfig
from datasets.data_factory import DataFactory
from model.model_factory import ModelFactory
from modules.metrics import sim_matrix_training, t2v_metrics
from transformers import CLIPTokenizer

def get_all_video_files(video_dir):
    """
    Get all video files in the directory.
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')  # Add more extensions if needed
    video_files = [os.path.join(video_dir, file) for file in os.listdir(video_dir) if file.endswith(video_extensions)]
    return video_files

def inference(config, query, video_dir):
    os.environ['TOKENIZERS_PARALLELISM'] = "false"

    # Set up GPU
    if config.gpu is not None and config.gpu != '99':
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if not torch.cuda.is_available():
            raise Exception('NO GPU!')

    # Set seed for reproducibility
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load tokenizer and model
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)
    model = ModelFactory.get_model(config)
    model.eval()

    # Load checkpoint if specified
    if config.load_epoch is not None:
        checkpoint_path = "model_best.pth" if config.load_epoch == -1 else f"checkpoint-epoch{config.load_epoch}.pth"
        checkpoint = torch.load(os.path.join(config.model_path, checkpoint_path))
        model.load_state_dict(checkpoint['state_dict'])
    
    # Get all video files in the directory
    video_paths = get_all_video_files(video_dir)
    
    # Prepare input data
    tokenized_query = tokenizer(query, return_tensors='pt', padding=True, truncation=True)
    video_data_loader = DataFactory.get_data_loader(config, split_type='test')

    # Move data to device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    tokenized_query = {key: val.to(device) for key, val in tokenized_query.items()}

    text_embeds_stochastic_allpairs = []

    with torch.no_grad():
        for batch_idx, data in enumerate(video_data_loader):
            # Move video data to device
            data['video'] = data['video'].to(device)

            # Model forward pass
            text_embeds, video_embeds_pooled, text_embeds_stochastic, _, _ = model(
                {'text': tokenized_query, 'video': data['video']}, 
                is_train=False
            )

            # Compute similarity
            sims = sim_matrix_training(text_embeds_stochastic, video_embeds_pooled, config.pooling_type)

            text_embeds_stochastic_allpairs.append(sims.cpu())

    # Concatenate all the similarity matrices
    final_sims = torch.cat(text_embeds_stochastic_allpairs, dim=0)
    
    # Compute metrics if necessary
    ranks = t2v_metrics(final_sims)
    print("Inference results (ranks):", ranks)

    return final_sims

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Add arguments from AllConfig
    parser.add_argument("--datetime", type=str, default=None)
    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--videos_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--noclip_lr", type=float, default=1e-5)
    parser.add_argument("--transformer_dropout", type=float, default=0.3)
    parser.add_argument("--dataset_name", type=str, default="MSRVTT")
    parser.add_argument("--msrvtt_train_file", type=str, default="9k")
    parser.add_argument("--stochasic_trials", type=int, default=20)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--load_epoch", type=int, default=0)
    parser.add_argument("--exp_name", type=str, required=True)
    
    # Add custom arguments for inference
    parser.add_argument("--query", type=str, required=True, help="The query text for inference.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing video files.")
    
    args = parser.parse_args()
    
    # Create a config object from the parsed arguments
    config = AllConfig()
    config.__dict__.update(vars(args))
    
    inference(config=config, query=args.query, video_dir=args.video_dir)
