# Update and install dependencies
sudo yum update -y
sudo yum install -y python3-pip
sudo yum install -y git

# Install Python dependencies
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip3 install transformers



python3 aws_sara_model.py --use_mo_sara --input_text "Hello, world!"