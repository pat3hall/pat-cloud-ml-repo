#!/bin/bash
apt update
apt install apache2 -y
systemctl enable apache2
systemctl start apache2
apt install unzip -y
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install
cd /var/www/html
echo "<html><body><h1>My AZ is" > index.html 
curl http://169.254.169.254/latest/meta-data/placement/availability-zone >> index.html
echo "<html><body><h1>My instance-id is    " >> index.html 
curl http://169.254.169.254/latest/meta-data/instance-id >> index.html
echo "<html><body><h1>My Public IP is      " >> index.html 
curl http://169.254.169.254/latest/meta-data/public-ipv4 >> index.html
echo "<html><body><h1>My Private IP is     " >> index.html 
curl http://169.254.169.254/latest/meta-data/local-ipv4 >> index.html
echo "</h1></body></html>" >> index.html 
