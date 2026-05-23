FROM rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.7.1 
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt