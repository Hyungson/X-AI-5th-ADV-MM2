import os
import torch
import random
import numpy as np
from config.all_config import AllConfig
from datasets.data_factory import DataFactory
from InferenceDataset import InferenceDataset
from torch.utils.data import DataLoader
from model.model_factory import ModelFactory
from modules.metrics import t2v_metrics, v2t_metrics
from modules.loss import LossFactory
from trainer.trainer_stochastic import Trainer
from config.all_config import gen_log
from torchvision import transforms


# @WJM: solve num_workers
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    # config
    config = AllConfig()
    '''print(config)  # config가 None이 아닌지 확인
    print(config.num_epochs)  # num_epochs가 올바르게 설정되었는지 확인'''
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    writer = None

    # GPU
    if config.gpu is not None and config.gpu != '99':
        print('set GPU')
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if not torch.cuda.is_available():
            raise Exception('NO GPU!')

    # @WJM: add log
    msg = f'model pth = {config.model_path}'
    gen_log(model_path=config.model_path, log_name='log_trntst', msg=msg)
    gen_log(model_path=config.model_path, log_name='log_trntst', msg='record all training and testing results')

    # seed
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # CLIP
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)

    # data I/O
    test_data_loader  = DataFactory.get_data_loader(config, split_type='test')
    model = ModelFactory.get_model(config)

    # metric
    if config.metric == 't2v':
        metrics = t2v_metrics
    elif config.metric == 'v2t':
        metrics = v2t_metrics
    else:
        raise NotImplemented

    loss = LossFactory.get_loss(config.loss)

    trainer = Trainer(model=model,
                      loss=loss,
                      metrics=metrics,
                      optimizer=None,
                      config=config,
                      train_data_loader=None,
                      valid_data_loader=test_data_loader,
                      lr_scheduler=None,
                      writer=writer,
                      tokenizer=tokenizer)
    
    
    # Load the best model
    trainer.load_bestmodel(model)
    
    # Define image transforms (ensure all images have consistent dimensions)
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize all frames to 224x224
        transforms.ToTensor(),  # Convert frames to tensor
    ])
    
    # Custom Inference Dataset 생성
    inference_dataset = InferenceDataset(config, img_transforms=img_transforms)
    inference_data_loader = DataLoader(inference_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # Trainer에서 inference_data_loader를 사용
    trainer.valid_data_loader = inference_data_loader

    prompt = "driving a car"  # "A man playing guitar in a park"
    best_video_id, best_similarity = trainer.inference(prompt)
    print(f"prompt: {prompt}")
    print(f"The video ID with the highest similarity is: {best_video_id}, Similarity = {best_similarity}")
    


if __name__ == '__main__':
    main()
    
# MSRVTT inference
# python inference.py --datetime=\data\MSRVTT   --arch=clip_stochastic   --videos_dir=data\MSRVTT\vids  --batch_size=32 --noclip_lr=3e-5 --transformer_dropout=0.3  --dataset_name=MSRVTT --msrvtt_train_file=9k   --stochasic_trials=20 --gpu='0'  --load_epoch=0  --num_epochs=5  --exp_name=MSR-VTT-9k

# Holywood2 inference
# python inference.py --datetime=\data\hollywood2   --arch=clip_stochastic   --videos_dir=data\hollywood2\test_clips  --batch_size=32 --noclip_lr=3e-5 --transformer_dropout=0.3  --dataset_name=hw2   --stochasic_trials=20 --gpu='0'  --load_epoch=0  --num_epochs=5  --exp_name=hw2
# 서버에서 돌릴 때
#python inference.py --datetime=./data/hollywood2   --arch=clip_stochastic   --videos_dir=./data/hollywood2/test_clips  --batch_size=16 --noclip_lr=3e-5 --transformer_dropout=0.3  --dataset_name=hw2   --stochasic_trials=20 --gpu='0'  --load_epoch=0  --num_epochs=5  --exp_name=hw2

