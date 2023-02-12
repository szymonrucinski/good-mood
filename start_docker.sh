docker build -f Dockerfile . -t "good-mood:latest"
docker run -p 8000:8000 good-mood:latest