#!/usr/bin/env python
# numpy01 image, see environment-numpy.org for construction
# (DL AMI v 3.0 based)
#
# us-east-1 AMIs
# numpy00: ami-f9d6dc83
# numpy01: ami-5b524f21

from collections import OrderedDict
import argparse
import os
import sys
import time

import boto3

module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')
import util
util.install_pdb_handler()

parser = argparse.ArgumentParser(description='launch')
parser.add_argument('--ami', type=str, default='ami-e580c79d',
                     help="name of AMI to use ")
parser.add_argument('--group', type=str, default='dawn_runs',
                     help="name of the current run")
parser.add_argument('--name', type=str, default='pytorch_test',
                     help="name of the current run")
parser.add_argument('--instance-type', type=str, default='p3.2xlarge',
                     help="type of instance")
parser.add_argument('--zone', type=str, default='us-west-2a',
                    help='which availability zone to use')
parser.add_argument('--linux-type', type=str, default='ubuntu',
                    help='which linux to use: ubuntu or amazon')
parser.add_argument('--role', type=str, default='launcher',
                    help='launcher or worker')
args = parser.parse_args()

def main():
  import aws_backend

  run = aws_backend.make_run(args.name, ami=args.ami,
                             availability_zone=args.zone,
                             linux_type=args.linux_type)
  job = run.make_job('pytorch', instance_type=args.instance_type)
  job.wait_until_ready()
  print(job.connect_instructions)
  

  # tensorboard stuff
  # if tensorboard is running, kill it, it will prevent efs logdir from being
  # deleted
#   job.run("tmux kill-session -t tb || echo ok")
#   logdir = '/efs/runs/%s/%s'%(args.group, args.name)
#   job.run('rm -Rf %s || echo failed' % (logdir,)) # delete prev logs
  
  # Launch tensorboard visualizer in separate tmux session
#   job.run("tmux new-session -s tb -n 0 -d")
#   job.run("tmux send-keys -t tb:0 'source activate pytorch_p36' Enter")
#   job.run("tmux send-keys -t tb:0 'tensorboard --logdir %s' Enter"%(logdir,))


#   run pytorch
  job.run('source activate pytorch_p36')
  job.run('killall python || echo failed')  # kill previous run
  job.upload('resnet.py')
  job.upload('fp16util.py')
  job.upload('distributed.py')
  job.upload('multiproc.py')
  job.upload('train_nv.py')
#   job.upload('train_nv.sh')
#   job.run_async('bash train_nv.sh --logdir=%s'%(logdir,))


if __name__=='__main__':
  main()
