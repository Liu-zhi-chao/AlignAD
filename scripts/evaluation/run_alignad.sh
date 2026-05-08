WORKER=ray_distributed
# WORKER=single_machine_thread_pool
TRAIN_TEST_SPLIT=navtest
CHECKPOINT='/home/data1/LiuZhichao/CodeSource/E2E/AlignDrive/exp/training_aligndrive_agent/91.76-02.01/lightning_logs/version_0/checkpoints/AlignAD_navsim.ckpt'
# Example: CUDA_VISIBLE_DEVICES=1 to use GPU 1, or CUDA_VISIBLE_DEVICES=0,1 to use GPU 0 and 1
export CUDA_VISIBLE_DEVICES=1
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_gpu.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=alignad_agent \
worker=$WORKER \
experiment_name=alignad_agent_eval \
agent.checkpoint_path=$CHECKPOINT