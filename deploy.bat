@echo off
echo === 배포 시작 ===
ssh -i "D:\pemkey\health.pem" ubuntu@ec2-32-236-113-89.ap-southeast-2.compute.amazonaws.com "cd /opt/medicine_similarity && git pull origin main && sudo systemctl restart medicine-api && sleep 2 && systemctl is-active medicine-api && echo 배포 완료"
echo === 완료 ===
pause
