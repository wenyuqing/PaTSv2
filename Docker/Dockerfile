FROM tangchuanxin/slowfast:deepspeed

#RUN #pip install pathlib ftfy termcolor regex pandas && \
#    pip install mmcv-full && \
#    pip install ftfy regex tqdm && \
#    pip install git+https://github.com/openai/CLIP.git


RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && \
    cd ..
