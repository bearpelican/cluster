#!/usr/bin/env python
# Tools for manipulating VPCs
import boto3
import sys
import os

import util as u

def _get_name(tags):
  """Helper utility to extract name out of tags dictionary.
      [{'Key': 'Name', 'Value': 'nexus'}] -> 'nexus'
 
     Assert fails if there's more than one name.
     Returns '' if there's less than one name.
  """
  
  names = [entry['Value'] for entry in tags if entry['Key']=='Name']
  if not names:
    return ''
  if len(names)>1:
    assert False, "have more than one name: "+str(names)
  return names[0]


def list_vpcs():
  for region in ['us-west-1', 'us-west-2', 'us-east-1']:
    print()
    print('='*80)
    print(region)
    print('='*80)
    client = boto3.client('ec2', region_name=region)
    ec2 = boto3.resource('ec2', region_name=region)
    response = client.describe_vpcs()

    for vpc_response in response['Vpcs']:
      key = _get_name(vpc_response.get('Tags', []))
      vpc = ec2.Vpc(vpc_response['VpcId'])
      print("%-16s %-16s" %(vpc.id, key))
      print('-'*40)
      subnets = list(vpc.subnets.all())
      if not subnets:
        print("<no subnets>")
      else:
        for subnet in subnets:
          print("%-16s %-16s"%(subnet.id, subnet.availability_zone))
      print()
      

def _create_ec2_client():
  REGION = u.get_region()
  return boto3.client('ec2', region_name=REGION)


def main():
  if len(sys.argv) < 2:
    mode = 'list'
  else:
    mode = sys.argv[1]

  if mode == 'list':
    list_vpcs()
  elif mode == 'delete':
    assert len(sys.argv) == 3
    
    assert 'AWS_DEFAULT_REGION' in os.environ
    client = u.create_ec2_client()
    ec2 = u.create_ec2_resource()
    response = client.describe_vpcs()
    for vpc_response in response['Vpcs']:
      vpc_name = _get_name(vpc_response.get('Tags', []))
      vpc = ec2.Vpc(vpc_response['VpcId'])
      if vpc_name == sys.argv[2] or vpc.id == sys.argv[2]:
        print("Deleting VPC name=%s, id=%s"%(vpc_name, vpc.id))
        
        for subnet in vpc.subnets.all():
          print("Deleting subnet %s" % (subnet.id))
          assert u.is_good_response(subnet.delete())

        for gateway in vpc.internet_gateways.all():
          print("Deleting gateway %s" % (gateway.id))
          assert u.is_good_response(gateway.detach_from_vpc(VpcId=vpc.id))
          assert u.is_good_response(gateway.delete())

        for security_group in vpc.security_groups.all():
          try:
            assert u.is_good_response(security_group.delete())
          except Exception as e:
            print("Failed with "+str(e))
            
        for route_table in vpc.route_tables.all():
          print("Deleting route table %s" % (route_table.id))
          try:
            assert u.is_good_response(route_table.delete())
          except Exception as e:
            print("Failed with "+str(e))
          
        if u.is_good_response(client.delete_vpc(VpcId=vpc.id)):
          print("Succeeded deleting VPC ", vpc.id)

if __name__=='__main__':
  main()
