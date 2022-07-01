# JMNET
# JMNET: Arbitrary-Shaped Scene Text Detection Using Multi-Space Perception

Install
```shell script
pip install -r requirement.txt
./compile.sh
```

## Training
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py ${CONFIG_FILE}
```
For example:
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py config/ifcsn/ifcsn_r18_ic15.py
```

## Test
```
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
```
For example:
```shell script
python test.py config/ifcsn/ifcsn_r18_ic15.py checkpoints/ifcsn_r18_ic15/checkpoint.pth.tar
```

## Speed
```shell script
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --report_speed
```
For example:
```shell script
python test.py config/ifcsn/ifcsn_r18_ic15.py checkpoints/ifcsn_r18_ic15/checkpoint.pth.tar --report_speed
```
