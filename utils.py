import functools
import requests
from contextlib import nullcontext

import wandb
# import ray
import hydra
from omegaconf import OmegaConf, DictConfig
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import transformers
from numerize.numerize import numerize



def create_wandb_wrapper(main_func, prefix=''):
    """
    main_func(conf, unique_name)
    """
    def wrapper(conf):
        project_name = conf.wandb.project

        # Not a clean solution
        conf_dict = OmegaConf.to_container(conf, resolve=True)
        if 'run_id' in conf.keys() and conf.run_id != "":
            training_config = retrieve_config(conf.wandb.entity, project_name, conf.run_id)
            conf_dict['training_config'] = training_config
        conf = OmegaConf.create(conf_dict)
        if 'wandb' in conf.keys() and conf.wandb.enabled:
            context = wandb.init(entity=conf.wandb.entity, project=project_name, config=conf_dict)
            context.name = prefix + context.name
            unique_name = context.name
        else:
            context = nullcontext()
            unique_name = 'a_unique_name'
        with context:
            out = main_func(conf, unique_name)
            if 'notify_slack' in conf.wandb.keys() and conf.wandb.notify_slack:
                context.alert(title="Task done", text=f"Run {unique_name} done!")

        # with wandb.init(entity=conf.wandb.entity, project=project_name, config=conf_dict) as run:
        #     run.name = prefix+run.name
        #     out = main_func(conf, run.name)

        return out
        # else:
        #     out = main_func(conf, 'a_unique_name')
        #     if 'notify_slack' in conf.wandb.keys() and conf.wandb.notify_slack:
        #         run.alert(title="Task done", text=f"Run {run.name} done!")
        #     return out

    return wrapper


def retrieve_config(entity, project, exp_id):
    api = wandb.Api()
    print(f'Retrieving config from the run <{exp_id}> ...')
    run = api.run(f"{entity}/{project}/{exp_id}")
    config = run.config
    config['wandb_exp_name'] = run.name
    return config

# def retrieve_config2(entity, project, exp_name):
#     api = wandb.Api()
#     print(f'Retrieving config from the run <{exp_id}> ...')
#     run = api.run(f"{entity}/{project}/{exp_id}")
#     config = run.config
#     return config


# def create_wandb_wrapper_with_prefix(main_func, prefix):
#     """
#     main_func(conf, unique_name)
#     """
#     def wrapper(conf):
#         project_name = conf.wandb.project
#         exp_name = f'{prefix}_rl_training'
#         with wandb.init(entity=conf.wandb.entity, project=project_name, config=OmegaConf.to_container(conf, resolve=True), name=exp_name) as run:
#             out = main_func(conf, run.name)
#
#             if 'notify_slack' in conf.wandb.keys() and conf.wandb.notify_slack:
#                 run.alert(title="Task done", text=f"Run {run.name} done!")
#         # print(f'DEBUG from wandb: {out}')
#         return out
#     return wrapper


# def create_ray_wrapper(main_func, num_cpus=15, num_gpus=1):
#     """
#     main_func(conf)
#     """
#     @ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)
#     def ray_wrapper(conf):
#             main_func(conf)
#
#     def my_wrapper(conf):
#         if 'ray' in conf.keys() and conf.ray.enabled:
#             ray.init(address=conf.ray.address)
#             out = ray.get(ray_wrapper.remote(conf))
#         else:
#             print('-- Running without RAY')
#             out = main_func(conf)
#         return out
#
#     return my_wrapper


def create_hydra_wrapper(main_func, path_to_conf_dir, conf_name):
    """
    This is the entry point.

    - main_func(conf: DictConfig)
    - path_to_config_dir: str
    - conf_name: str
    """
    @hydra.main(version_base=None, config_path=path_to_conf_dir, config_name=conf_name)
    def hydra_wrapper(conf: DictConfig):
        print(OmegaConf.to_yaml(conf))
        out = main_func(conf)

        # print(f'DEBUG from hydra: {out}')
        return out

    return hydra_wrapper


def plot_distribution(x: Tensor, title: str) -> plt.Figure:
    x = x.detach().cpu()
    fig, axes = plt.subplots(2, 1, constrained_layout=True)
    axes = axes.flatten()

    hist, edges = torch.histogram(x, density=True)
    axes[0].plot(edges[:-1], hist)

    mean, std = x.mean(), x.std()
    fig.suptitle(f"{title} | Mean: {mean:.4f} Std: {std:.4f}")
    fig.supxlabel("X")
    fig.supylabel("Density")

    return fig


def my_generation(model, tokenizer, texts, max_input_length=8, num_return_sequences=1,
        max_output_length=10, do_sample=False, include_input=True, device='cuda'):
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    generation_kwargs = {
        "min_length": -1,
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": max_output_length,
        'num_return_sequences': num_return_sequences,
    }
    query_tensors = tokenizer(texts, max_length=max_input_length, padding='max_length',
            truncation=True)
    query_tensors['input_ids'] = torch.tensor(query_tensors['input_ids']).to(device)
    query_tensors['attention_mask'] = torch.tensor(query_tensors['attention_mask']).to(device)
    resp_tensor = model.generate(**query_tensors, **generation_kwargs)
    if not include_input:
        resp_tensor = resp_tensor[:, max_input_length:]
    resps_text = tokenizer.batch_decode(resp_tensor, skip_special_tokens=True)
    return resps_text


def load_tokenizer(tokenizer_name_or_path, model_name_or_path, cache_dir):
    tokenizer_name_or_path = tokenizer_name_or_path or model_name_or_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path,\
            cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def save_hf_format(policy, tokenizer, dir_path):
    policy.save_pretrained(dir_path)
    tokenizer.save_pretrained(dir_path)


def model_summary(model: torch.nn.Module):
    print(model)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in model.parameters())
    print()
    print('|----------------------------------')
    print('| MODEL SUMMARY: ')
    print(f'| #params: {numerize(num_params)}')
    print(f'| #trainable_params: {numerize(num_trainable_params)}')
    print('|----------------------------------')
    print()

