for THE_CT in 2.0
do
    for THE_INIT in 1 
    do
        for THE_LR in 0.001
        do
            CUDA_VISIBLE_DEVICES=3 python main.py --phase=adapt_train --setname=ACON_Ag --ct_wgt=${THE_CT} --adapt_init=${THE_INIT} --shot=10 --lr=${THE_LR}
        done
    done
done

# THE_CT: 0.5 1.0 2.0
# THE_LR: 0.0001 0.001