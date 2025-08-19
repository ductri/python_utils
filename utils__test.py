from dataclasses import dataclass
from utils import create_wandb_wrapper



def main(conf, unique_name):
    print(unique_name)

if __name__ == "__main__":
    runner = create_wandb_wrapper(main)
    runner(None)

