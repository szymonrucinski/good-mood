FROM python:3.10
# Create api directory
WORKDIR /usr/src/good-mood
# Install app dependencies
ADD ./ /usr/src/good-mood
RUN ls /usr/src/good-mood
RUN  pip install --upgrade pip \
     && pip install pipenv \
     && pipenv install --skip-lock \
     && pipenv run pip install .

EXPOSE 8000

CMD ["pipenv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
