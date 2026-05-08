python navsim/planning/script/run_dataset_caching.py \
agent=alignad_agent \
agent.config.cache_data=True \
experiment_name=training_cacheing_alignad_agent \
train_test_split=navtrain \
cache_path=$NAVSIM_EXP_ROOT/navsim_cache