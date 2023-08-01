# (вдохновлялся https://gist.github.com/jprjr/7667947?permalink_comment_id=3684823#gistcomment-3684823)
# образ для контейнера
FROM nvcr.io/nvidia/tensorrt:22.08-py3

ENV PYTHON_VERSION 3.10.10

#Set of all dependencies needed for pyenv to work on Ubuntu
RUN apt-get update && apt-get install -y --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget ca-certificates curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev mecab-ipadic-utf8 git

# Set-up necessary Env vars for PyEnv
ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# Install pyenv
RUN set -ex \
    && curl https://pyenv.run | bash \
    && pyenv update \
    && pyenv install $PYTHON_VERSION \
    && pyenv global $PYTHON_VERSION \
    && pyenv rehash

# копируем pyproject.toml нашего проекта
COPY pyproject.toml /code/
COPY poetry.lock /code/

WORKDIR /code

RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev --no-interaction --no-ansi --no-root

# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# копируем проект
COPY hello.py /code/
COPY pages /code/pages
COPY example_images /code/example_images
COPY test_video /code/test_video

COPY RetinaFace_weights /code/RetinaFace_weights
COPY models /code/models
COPY model /code/model
RUN wget -P /root/.cache/torch/hub/checkpoints \ 
    https://download.pytorch.org/models/resnet50-0676ba61.pth

CMD ["python", "-m", "streamlit", "run", "hello.py"]