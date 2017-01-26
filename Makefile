install_venv:
	# Install a virtual environment
	virtualenv tf_cpu
	# Activate the virutal environment
	source tf_cpu/bin/activate
	# Install tensorflow
	export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp27-none-linux_x86_64.whl
	pip install $TF_BINARY_URL

install_no_venv:
	# Install tensorflow
	export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp27-none-linux_x86_64.whl
	pip install --user $TF_BINARY_URL

install_conda:
	conda install -c conda-forge tensorflow
