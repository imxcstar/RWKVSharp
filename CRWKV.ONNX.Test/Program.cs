using CLLM.Core;
using RWKV;

var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Model");
var models = Directory.GetFiles(path, "*.onnx").Select(x => Path.GetFileName(x)).ToList();
if (models.Count == 0)
{
    Console.WriteLine($"Model folder No model, please go to \"https://huggingface.co/imxcstar/rwkv-4-raven-onnx/tree/main\" Download model to \"{path}\"");
    return;
}

Console.WriteLine($"Select Model:");
for (int i = 0; i < models.Count; i++)
{
    Console.WriteLine($"\t{i}: {models[i]}");
}

selectModel:
Console.Write("Input Model Number: ");
var modelNumberStr = Console.ReadLine();
if (string.IsNullOrWhiteSpace(modelNumberStr))
    goto selectModel;
var modelNumber = int.Parse(modelNumberStr);
if (modelNumber < 0 || modelNumber >= models.Count)
    goto selectModel;

var modelName = models[modelNumber];

var modelNames = modelName.Split("_");
var n_layer = int.Parse(modelNames[1]);
var n_embd = int.Parse(modelNames[2]);

Console.WriteLine($"Loading...");

var rf = new RunnerFactory();
var tokenizerFileName = "20B_tokenizer.json";
if (modelName.ToLower().Contains("world"))
    tokenizerFileName = "rwkv_vocab_v20230424.txt";
rf.RegisterRWKVOnnxModel(Path.Combine(path, modelName), Path.Combine(path, tokenizerFileName), n_embd, n_layer);

var r = rf.Builder();

while (true)
{
    Console.Write(">");
    var value = Console.ReadLine();
    if (string.IsNullOrEmpty(value))
        continue;
    r.Run(value.Replace("\\r\\n", "\r\n").Replace("\\r", "\r").Replace("\\n", "\n"), v =>
    {
        Console.Write(v);
    });
    Console.WriteLine();
}