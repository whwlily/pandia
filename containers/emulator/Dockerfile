FROM johnson163/pandia_sender_base

# 更新包管理器并安装 jq
RUN apt update && apt install -y jq

COPY . /app/
WORKDIR /app

CMD python -m sb3_client
