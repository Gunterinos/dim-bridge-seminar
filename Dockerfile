FROM python
WORKDIR /dim-bridge
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-u", "app.py"]

