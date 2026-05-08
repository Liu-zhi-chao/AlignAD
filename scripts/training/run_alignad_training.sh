TRAIN_TEST_SPLIT=navtrain

HYDRA_FULL_ERROR=1 python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=alignad_agent \
experiment_name=training_alignad_agent \
train_test_split=$TRAIN_TEST_SPLIT \
use_cache_without_dataset=True \
force_cache_computation=False   \
agent.lr=1e-4   \
dataloader.params.batch_size=16  \
# trainer.params.devices=1    \