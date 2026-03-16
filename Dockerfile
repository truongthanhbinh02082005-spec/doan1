FROM python:3.9

WORKDIR /code

# Cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code và dữ liệu vào
COPY . .

# Cấp quyền để server ghi file (lưu biểu đồ vào thư mục static)
RUN chmod -R 777 /code

# Chạy backend (phải khớp với tên file back_end.py của ông)
CMD ["python", "back_end.py"]
