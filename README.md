# chainer-SeqGAN
 
- implementation of [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/abs/1609.05473)
- Complete oracle test in this paper

## requirements

- Python > 3.4
- Chainer > 1.5
- Tensorflow (CPU-only)
```
# Ubuntu/Linux 64-bit Python 3.4
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0-cp34-cp34m-linux_x86_64.whl

# Ubuntu/Linux 64-bit Python 3.5
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0-cp35-cp35m-linux_x86_64.whl

# Mac OS X, CPU only, Python 3.4 or 3.5:
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0-py3-none-any.whl

pip install $TF_BINARY_URL
```

## Usage

```
cd oracle_test && python run_sequence_gan.py 
```


Any advice or suggestion is strongly welcomed in issues.
