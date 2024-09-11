## Support RWKV4 Raven/World RWKV5 / RWKV6 World/Finch 1B5-14B (ggml) (CPU/GPU)

### Use
Development Environment: Windows, Visual Studio 2022

```
git clone https://github.com/imxcstar/CSharp-RWKV.git

cd CSharp-RWKV

git submodule update --progress --init --remote

git submodule update --progress --init --recursive
```

Download model from [Hugging Face](https://huggingface.co/BlinkDL)

Open CRWKV.sln

Compile the CRWKV.GGML.Convert project and then use this project to convert the downloaded model.

Compile the CRWKV.GGML.Test project, and then put the converted model model into the Model directory in the project.

## Model placement location

![3.png](/Preview/3.png)

## Preview

![1.png](/Preview/1.png)

![2.png](/Preview/2.png)

## Reference Links

* **RWKV Hugging Face repo**: https://huggingface.co/RWKV
* **rwkv.cpp**: https://github.com/RWKV/rwkv.cpp
* **RWKV**: https://github.com/BlinkDL/RWKV-LM
* **ChatRWKV**: https://github.com/BlinkDL/ChatRWKV
* **Tokenizer**: https://github.com/Alex1911-Jiang/GPT-3-Encoder-Sharp
