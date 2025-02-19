### English | [中文](README.ZH.md)
## Based on [rwkv.cpp](https://github.com/RWKV/rwkv.cpp), supports running RWKV4, RWKV5, RWKV6, RWKV7 World/Raven/Finch 1B5-14B (ggml) (CPU/GPU)
### Installation
Install from [nuget.org](https://www.nuget.org/packages?q=RWKVSharp)
```
PM> Install-Package RWKVSharp
```
If you are running on Windows CPU, you also need to install the following:
```
PM> Install-Package RWKVSharp.Native.Cpu.avx2.win-x64
```
For native libraries on other systems, you can search at [nuget.org](https://www.nuget.org/packages?q=RWKVSharp)

### How to use RWKVSharp
```csharp
using RWKVSharp.Core;
using RWKVSharp;
var rf = new RunnerFactory();
rf.RegisterRWKVGGMLModel("RWKV-x070-World-0.1B-v2.8-20241210-ctx4096-FP16.bin", "rwkv_vocab_v20230424.txt");
// RWKV-x070-World-0.1B-v2.8-20241210-ctx4096-FP16.bin conversion instructions are detailed below.
// rwkv_vocab_v20230424.txt can be downloaded from https://github.com/imxcstar/RWKVSharp/tree/main/RWKVSharp.Test/Model
var r = rf.Builder();
while (true)
{
    Console.Write(">");
    var value = Console.ReadLine();
    if (string.IsNullOrEmpty(value))
        continue;
    r.Generate(value.Replace("\\r\\n", "\r\n").Replace("\\r", "\r").Replace("\\n", "\n"), Console.Write);
    Console.WriteLine();
}
```

### Model conversion method
1. First, download the `.pth` model you wish to use from https://huggingface.co/RWKV
2. Then, download the `RWKVSharp.Convert.zip` conversion tool from [Releases](https://github.com/imxcstar/RWKVSharp/releases) (currently only supports Windows)
3. Use the conversion tool to convert the model.
![4.png](/Preview/4.png)
4. After conversion, you'll obtain a `.bin` model file that can be used with RWKVSharp.
![5.png](/Preview/5.png)

### Model Quantization Tool
You can download `RWKVSharp.Quantize.zip` from [Releases](https://github.com/imxcstar/RWKVSharp/releases) (currently only supports Windows).

## Preview
![1.png](/Preview/1.png)
![2.png](/Preview/2.png)

## Links
* **RWKV Hugging Face repository**: https://huggingface.co/RWKV
* **rwkv.cpp**: https://github.com/RWKV/rwkv.cpp
* **RWKV**: https://github.com/BlinkDL/RWKV-LM
* **ChatRWKV**: https://github.com/BlinkDL/ChatRWKV
* **Tokenizer**: https://github.com/Alex1911-Jiang/GPT-3-Encoder-Sharp