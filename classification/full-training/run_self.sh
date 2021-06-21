ARG_XONLY=0
ARG_INIT=1

set -x

for ARG_PRELR in 0.0005
do
    CUDA_VISIBLE_DEVICES=3 python main.py --phase=self_train --setname=BTNx --lr=${ARG_PRELR} --xonly=${ARG_XONLY} --self_init=${ARG_INIT}
done