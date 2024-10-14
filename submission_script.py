'''Script to submit the training job for the benchamrk model.'''
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role
from sagemaker.debugger import Rule, DebuggerHookConfig, CollectionConfig, rule_configs
from sagemaker.debugger import ProfilerRule, ProfilerConfig, FrameworkProfile

role = get_execution_role()
# Set hyperparameters
hyperparameters = {"epochs": "10",
                "batch_size": "128",
                "learning_rate": "0.01"}

# Create profiling and debugging rules
rules = [
    Rule.sagemaker(rule_configs.loss_not_decreasing(),
                      rule_parameters={
                          "max_patience": "10",
                          "threshold": "0.01"}),
    Rule.sagemaker(rule_configs.vanishing_gradient()),
    ProfilerRule.sagemaker(rule_configs.LowGPUUtilization()),
    ProfilerRule.sagemaker(rule_configs.CPUBottleneck()),
    ProfilerRule.sagemaker(rule_configs.MaxInitializationTime()),
    ProfilerRule.sagemaker(rule_configs.ProfilerReport()),
]

# Set profiler and debugger configs
#debugger_config = DebuggerHookConfig(
#    collection_configs=[
#        CollectionConfig(name="losses", parameters={"save_interval": "5"})
#    ])
debugger_config = DebuggerHookConfig(
    hook_parameters={"train.save_interval": "100", "eval.save_interval": "10"}
)

profiler_config = ProfilerConfig(
    system_monitor_interval_millis=1000,
    framework_profile_params=FrameworkProfile(num_steps=5)
)

# Create the estimator
estimator = PyTorch(
    entry_point="train_benchmark.py",
    base_job_name="benchmark-job",
    role=role,
    instance_type="ml.g4dn.xlarge", #ml.g4dn.xlarge, ml.m5.xlarge
    instance_count=1,
    source_dir="./scripts",
    dependencies=["./scripts/requirements.txt"],
    hyperparameters=hyperparameters,
    debugger_hook_config=debugger_config, #Uncomment these lines to perform profiling and debug
    profiler_config=profiler_config, 
    rules=rules,
    framework_version="2.2", #Pytorch version List of supported versions: https://github.com/aws/deep-learning-containers/blob/master/available_images.md.
    py_version="py310"
)

# Set the input channels usig the URIs from the uplpoad
inputs = {'training': 's3://capstone520/data/train/ecoli_train_im.csv',
          'test': 's3://capstone520/data/test/ecoli_test_im.csv'}

# Launch the training job
estimator.fit(inputs, wait=True)
