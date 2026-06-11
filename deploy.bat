@echo off
echo === 배포 시작 ===
if "%EC2_SSH_KEY%"=="" (
  echo EC2_SSH_KEY 환경변수를 설정하세요.
  exit /b 1
)
if "%EC2_HOST%"=="" (
  echo EC2_HOST 환경변수를 설정하세요.
  exit /b 1
)
if "%EC2_USER%"=="" set EC2_USER=ubuntu
if "%EC2_APP_DIR%"=="" set EC2_APP_DIR=/opt/medicine_similarity
if "%EC2_SERVICE%"=="" set EC2_SERVICE=medicine-api
ssh -i "%EC2_SSH_KEY%" %EC2_USER%@%EC2_HOST% "cd %EC2_APP_DIR% && git pull origin main && sudo systemctl restart %EC2_SERVICE% && sleep 2 && systemctl is-active %EC2_SERVICE% && echo 배포 완료"
echo === 완료 ===
pause
