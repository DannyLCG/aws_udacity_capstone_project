'''Script to submit the training job for the benchamrk model.'''
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role
from sagemaker.debugger import ProfilerRule, rule_configs
from sagemaker.debugger import ProfilerConfig, FrameworkProfile

role = get_execution_role()
# Set hyperparameters
hyperparameters = {"epochs": "10",
                "batch-size": "128",
                "learning_rate": "0.015"}

# Create profiling and debugging rules
rules = [
    ProfilerRule.sagemaker(rule_configs.LowGPUUtilization()),
    ProfilerRule.sagemaker(rule_configs.CPUBottleneck()),
    ProfilerRule.sagemaker(rule_configs.MaxInitializationTime),
    ProfilerRule.sagemaker(rule_configs.ProfilerReport()),
]

# Set profiler and debugger configs
profiler_config = ProfilerConfig(
    system_monitor_interval_millis=500,
    framework_profile_params=FrameworkProfile(num_steps=10) #i)?
)

# Create the estimator
estimator = PyTorch(
    entry_point="scripts/train_benchmark.py",
    base_job_name="sagemaker-script-mode",
    role=role,
    instance_type="ml.g4dn.xlarge",
    instane_count=2,
    hyperparameters=hyperparameters,
    framework_version="2.2", #Pytorch version List of supported versions: https://github.com/aws/deep-learning-containers/blob/master/available_images.md.
    py_version="py310"
)

# Set the input channels usig the URIs from the uplpoad
inputs = {'training': 's3://capstone4910/data/train/ecoli_train_im.csv',
 'test': 's3://capstone4910/data/test/ecoli_test_im.csv'}

# Launch the training job
estimator.fit(inputs, wait=True)
