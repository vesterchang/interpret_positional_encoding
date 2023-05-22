# !/usr/bin/env bash
CUDA_ID=0

DATA_PATH="/workspace/data-bin/wmt14_en_de"
USER_DIR="sin_pe"
EXPERIMENT="ex_sin_pe_ende"

mkdir -p ${EXPERIMENT}/ckp ${EXPERIMENT}/gen ${EXPERIMENT}/tensorboard_log


# Training
CUDA_VISIBLE_DEVICES=${CUDA_ID} fairseq-train $DATA_PATH --user-dir ${USER_DIR} \
	--fp16 --left-pad-source --arch transformer_sin_pe_wmt_en_de \
	--patience 30 --max-update 150000 --max-tokens 4096 --update-freq 8 \
	--share-all-embeddings --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 \
	--lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --lr 7e-4 --warmup-updates 4000 --stop-min-lr 1e-9 \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
	--no-progress-bar --log-format json --log-interval 10 --keep-interval-updates 10 \
	--save-dir $EXPERIMENT/ckp --tensorboard-logdir $EXPERIMENT/tensorboard_log > $EXPERIMENT/exp.log


# Evaluating 

# average checkpoints
ckp_name="avg_update_last5"

python scripts/average_checkpoints.py --input ${EXPERIMENT}/ckp --num-update-checkpoints 5 --output ${EXPERIMENT}/ckp/${ckp_name}.pt

CUDA_VISIBLE_DEVICES=${CUDA_ID} fairseq-generate $DATA_PATH --path ${EXPERIMENT}/ckp/$ckp \
    --batch-size 128 --beam 4 --lenpen 0.6 --max-len-a 1 --max-len-b 50 --remove-bpe \
    --fp16 --left-pad-source --user-dir ${USER_DIR} > ${EXPERIMENT}/gen/${ckp_name}_gen.out 

grep ^H ${EXPERIMENT}/gen/${ckp_name}_gen.out | cut -f3- > ${EXPERIMENT}/gen/${ckp_name}_gen.out.sys
grep ^T ${EXPERIMENT}/gen/${ckp_name}_gen.out | cut -f2- > ${EXPERIMENT}/gen/${ckp_name}_gen.out.ref

bash scripts/compound_split_bleu.sh ${EXPERIMENT}/gen/${ckp_name}_gen.out


