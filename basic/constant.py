import os
import logging
import torch

ROOT_PATH = os.path.join(os.environ['HOME'], 'insCar')
device = torch.device("cuda:0")

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S',
    level=logging.INFO)
# logger.setLevel(logging.INFO)
