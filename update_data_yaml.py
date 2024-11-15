import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--yaml')
parser.add_argument('--train')
parser.add_argument('--test')

args = parser.parse_args()

with open(args.yaml, 'r') as f:
    y = yaml.safe_load(f)
    y['train'] = args.train
    y['val'] = args.test
    y['test'] = args.test

with open(args.yaml, 'w') as f:
    f.write(yaml.dump(y, default_flow_style=False, sort_keys=False))