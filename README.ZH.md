### [English](README.md) | 中文

## 基于 [rwkv.cpp](https://github.com/RWKV/rwkv.cpp), 支持运行 RWKV4, RWKV5, RWKV6, RWKV7 World/Raven/Finch 1B5-14B (ggml) (CPU/GPU)

### 安装

从 [nuget.org](https://www.nuget.org/packages?q=RWKVSharp) 安装
```
PM> Install-Package RWKVSharp
```

如果使用Windows CPU运行的话，还需要安装

```
PM> Install-Package RWKVSharp.Native.Cpu.avx2.win-x64
```

其它系统的原生库，可以在 [nuget.org](https://www.nuget.org/packages?q=RWKVSharp) 查询

### RWKVSharp使用方法
```csharp
using RWKVSharp.Core;
using RWKVSharp;

var rf = new RunnerFactory();
rf.RegisterRWKVGGMLModel("RWKV-x070-World-0.1B-v2.8-20241210-ctx4096-FP16.bin", "rwkv_vocab_v20230424.txt");

//RWKV-x070-World-0.1B-v2.8-20241210-ctx4096-FP16.bin 转换方法请看下面的说明
//rwkv_vocab_v20230424.txt 在 https://github.com/imxcstar/RWKVSharp/tree/main/RWKVSharp.Test/Model 可以下载

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

### 模型转换方法
1. 先从 https://huggingface.co/RWKV 下载你想运行的`.pth`模型

2. 然后在 [Releases](https://github.com/imxcstar/RWKVSharp/releases) 下载 `RWKVSharp.Convert.zip` 转换工具(暂时只支持Windows)

3. 然后使用转换工具对模型进行转换
![4.png](/Preview/4.png)

4. 得出`.bin`模型文件即可使用RWKVSharp进行调用
![5.png](/Preview/5.png)

### 模型量化工具
可以在 [Releases](https://github.com/imxcstar/RWKVSharp/releases) 下载 `RWKVSharp.Quantize.zip` (暂时只支持Windows)

## 预览

![1.png](/Preview/1.png)

![2.png](/Preview/2.png)

## 链接

* **RWKV Hugging Face repo**: https://huggingface.co/RWKV
* **rwkv.cpp**: https://github.com/RWKV/rwkv.cpp
* **RWKV**: https://github.com/BlinkDL/RWKV-LM
* **ChatRWKV**: https://github.com/BlinkDL/ChatRWKV
* **Tokenizer**: https://github.com/Alex1911-Jiang/GPT-3-Encoder-Sharp
