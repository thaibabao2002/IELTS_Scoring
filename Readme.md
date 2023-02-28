Tạo thư mục save_model
Tải 2 model từ link này về:
https://drive.google.com/drive/folders/1LuvkBTK7UUVZkwfYmdAV5QtFGkgWaeAG?usp=sharing
Chạy lệnh:
docker-compose up
hoặc 2 lệnh:
docker build -t inference:latest .
docker run -p 8000:8000 --name inference_container inference:latest
vào link để chạy thử API:
http://localhost:8000/docs
