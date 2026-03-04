import argparse
from yacs.config import CfgNode

_C = CfgNode(new_allowed=True)


parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=str, default=None)
parser.add_argument("--exp_name", type=str, default=None)
args = parser.parse_args()

cfg = _C.clone()
cfg.merge_from_file('configs/default.yml')
if args.scene is not None:
    cfg.scene = args.scene
if args.exp_name is not None:
    cfg.exp_name = args.exp_name