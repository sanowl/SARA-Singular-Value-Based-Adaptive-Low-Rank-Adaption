import boto3
import time

# AWS configuration
AMI_ID = 'ami-0cff7528ff583bf9a'  # Amazon Linux 2 AMI (HVM) - Kernel 5.10, SSD Volume Type
INSTANCE_TYPE = 't2.large'  # Adjust based on your needs
KEY_NAME = 'your-key-pair-name'  # Replace with your key pair name
SECURITY_GROUP_ID = 'sg-xxxxxxxxxxxxxxxxx'  # Replace with your security group ID

# Create EC2 client
ec2 = boto3.client('ec2')

def create_ec2_instance():
    instances = ec2.run_instances(
        ImageId=AMI_ID,
        InstanceType=INSTANCE_TYPE,
        KeyName=KEY_NAME,
        MinCount=1,
        MaxCount=1,
        SecurityGroupIds=[SECURITY_GROUP_ID],
        UserData=open('setup_and_run.sh').read()
    )
    return instances['Instances'][0]['InstanceId']

def wait_for_instance(instance_id):
    waiter = ec2.get_waiter('instance_running')
    waiter.wait(InstanceIds=[instance_id])
    time.sleep(60)  # Wait an additional 60 seconds for UserData script to complete

def get_public_ip(instance_id):
    response = ec2.describe_instances(InstanceIds=[instance_id])
    return response['Reservations'][0]['Instances'][0]['PublicIpAddress']

def main():
    print("Creating EC2 instance...")
    instance_id = create_ec2_instance()
    print(f"Instance created with ID: {instance_id}")

    print("Waiting for instance to be ready...")
    wait_for_instance(instance_id)

    public_ip = get_public_ip(instance_id)
    print(f"Instance is ready. Public IP: {public_ip}")
    print("You can now SSH into the instance and check the output.")

if __name__ == "__main__":
    main()