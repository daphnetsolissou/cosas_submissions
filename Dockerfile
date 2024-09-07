FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PROJECTDIR=/opt/app
WORKDIR $PROJECTDIR

RUN pip3 install openmim ftfy SimpleITK transformers numpy albumentations
RUN mim install mmengine mmcv==2.1.0
RUN pip3 install mmsegmentation==1.2.2

COPY ensemble_inference.py $PROJECTDIR
COPY dinov2_adapter.py $PROJECTDIR
COPY dinov2LinearProb.py $PROJECTDIR
#COPY model/best_val_mcScore_iter_68000_ce_loss_fold1.pth $PROJECTDIR
COPY config.py $PROJECTDIR


RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

LABEL authors="daphneTsolissou"
ENTRYPOINT ["python3", "ensemble_inference.py"]
#CMD ["/bin/bash"]