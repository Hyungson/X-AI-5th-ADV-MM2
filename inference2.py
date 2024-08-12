import os
import torch
import random
import numpy as np
from config.all_config import AllConfig
from datasets.data_factory import DataFactory
from model.model_factory import ModelFactory
from modules.metrics import sim_matrix_training, t2v_metrics
from transformers import CLIPTokenizer

def get_all_video_files(video_dir):

    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [os.path.join(video_dir, file) for file in os.listdir(video_dir) if file.endswith(video_extensions)]
    return video_files

def inference(query, video_dir):
    config = AllConfig()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"

    if config.gpu is not None and config.gpu != '99':
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if not torch.cuda.is_available():
            raise Exception('NO GPU!')

    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)
    model = ModelFactory.get_model(config)
    model.eval()

    checkpoint_path = os.path.join(config.model_path, "model_best.pth")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    video_paths = get_all_video_files(video_dir)
    
    tokenized_query = tokenizer(query, return_tensors='pt', padding=True, truncation=True)
    video_data_loader = DataFactory.get_data_loader(config, split_type='test')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    tokenized_query = {key: val.to(device) for key, val in tokenized_query.items()}

    text_embeds_stochastic_allpairs = []

    with torch.no_grad():
        for batch_idx, data in enumerate(video_data_loader):
            data['video'] = data['video'].to(device)

            text_embeds, video_embeds_pooled, text_embeds_stochastic, _, _ = model(
                {'text': tokenized_query, 'video': data['video']}, 
                is_train=False
            )

            sims = sim_matrix_training(text_embeds_stochastic, video_embeds_pooled, config.pooling_type)

            text_embeds_stochastic_allpairs.append(sims.cpu())

    final_sims = torch.cat(text_embeds_stochastic_allpairs, dim=0)
    
    ranks = t2v_metrics(final_sims)
    print("Inference results (ranks):", ranks)

    return final_sims

if __name__ == "__main__":
    query = "A man is driving a car."
    video_dir = "./videos2"
    inference(query, video_dir)