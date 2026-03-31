FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN touch src/__init__.py

CMD ["python", "-m", "src.train", "--model", "patchtst", "--data_dir", "data/raw", "--train_depts", "01,38,73", "--val_depts", "74", "--test_depts", "69", "--train_end", "2018-12-01", "--val_end", "2020-12-01", "--history_len", "24", "--horizon", "12", "--epochs", "10", "--batch_size", "64"]