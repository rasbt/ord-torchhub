## Prepare environment for training

```
conda create -n ord-torchhub python=3.8
conda activate ord-torchhub
pip install -r requirements.txt
```

## Download Dataset

```bash
git clone https://github.com/afad-dataset/tarball.git
cd tarball
cat AFAD-Full.tar.xz* > AFAD-Full.tar.xz
tar -xf AFAD-Full.tar.xz
```

## Run training

CORN:

```python
python resnet34_corn_afad.py \
--numworkers 3 \
--learningrate 0.0005 \
--seed 0 \
--cuda 0 \
--batchsize 16 \
--epochs 50 \
--overwrite true \
--output_dir ./resnet34_corn_afad_out
```

CORAL:

```python
python resnet34_coral_afad.py \
--numworkers 3 \
--learningrate 0.0005 \
--seed 0 \
--cuda 0 \
--batchsize 256 \
--epochs 50 \
--overwrite true \
--output_dir ./resnet34_coral_afad_out
```

Niu et al.:

```python
python resnet34_niu_afad.py \
--numworkers 3 \
--learningrate 0.0005 \
--seed 0 \
--cuda 0 \
--batchsize 256 \
--epochs 50 \
--overwrite true \
--output_dir ./resnet34_niu_afad_out
```

Regular cross entropy:

```python
python resnet34_crossentr_afad.py \
--numworkers 3 \
--learningrate 0.0005 \
--seed 0 \
--cuda 0 \
--batchsize 256 \
--epochs 50 \
--overwrite true \
--output_dir ./resnet34_crossentr_afad_out
```