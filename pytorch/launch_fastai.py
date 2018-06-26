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
# parser.add_argument('--instance-type', type=str, default='p3.2xlarge',
parser.add_argument('--instance-type', type=str, default='t2.large',
                     help="type of instance")
parser.add_argument('--zone', type=str, default='us-west-2a',
                    help='which availability zone to use')
parser.add_argument('--linux-type', type=str, default='ubuntu',
                    help='which linux to use: ubuntu or amazon')
parser.add_argument('--role', type=str, default='launcher',
                    help='launcher or worker')
args = parser.parse_args()


def attach_imagnet_ebs(aws_instance, job, tag='imagenet_high_perf'):
  ec2 = util.create_ec2_resource()
  v = list(ec2.volumes.filter(Filters=[{'Name':'tag:Name', 'Values':['imagenet_high_perf']}]).all())[0]
  if v.state != 'available': v.detach_from_instance()
  v.attach_to_instance(InstanceId=aws_instance.id, Device='/dev/xvdf')
  job.run('sudo mkdir mount_point')
  job.run('sudo mount /dev/xvdf mount_point')
  

def main():
  import aws_backend

  run = aws_backend.make_run(args.name, ami=args.ami,
                             availability_zone=args.zone,
                             linux_type=args.linux_type)
  job = run.make_job('pytorch', instance_type=args.instance_type)
  job.wait_until_ready()
  print(job.connect_instructions)

#   attach_imagnet_ebs(run.instances[0], job)
  

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

  # upload files
  job.upload('resnet.py')
  job.upload('distributed.py')
  job.upload('multiproc.py')
  job.upload('train_imagenet_fastai.py')
  job.upload('setup_env_fastai.sh')

  # run setup script
  job.run('./setup_env_fastai.sh')

  # run training
  # https://github.com/stanford-futuredata/dawn-bench-entries/pull/44/files
  job.run_async('python -m multiproc train_imagenet_fastai.py ~/mount_point/imagenet  --sz 224 -b 192 -j 8 --fp16 -a resnet50 --lr 0.40 --epochs 45 --small')
# time python -m multiproc fastai_imagenet.py $DATA_DIR --sz $SIZE -j 8 --fp16 -b $BS --loss-scale 512 --save-dir $SAVE_DIR $SARGS |& tee -a $SAVE_DIR/output.log
# python run_script.py -zone us-west-2b --launch-method find -iname p36 -p rn50_40_45_bnf_main_sml --run-script upload_scripts/train2.sh -sargs "-sargs '-a resnet50 --lr 0.40 --epochs 45 --small'"

if __name__=='__main__':
  main()
