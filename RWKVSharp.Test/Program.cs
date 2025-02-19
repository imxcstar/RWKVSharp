using RWKVSharp.Core;
using RWKVSharp;

var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Model");
var models = Directory.GetFiles(path).Where(x => ".bin".Contains(Path.GetExtension(x))).Select(x => Path.GetFileName(x)).ToList();
if (models.Count == 0)
{
    Console.WriteLine($"Model folder No model, please go to \"https://huggingface.co/imxcstar/rwkv-4-raven-ggml/tree/main\" Download model to \"{path}\"");
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

Console.WriteLine($"Loading...");

var rf = new RunnerFactory();
var tokenizerFileName = "20B_tokenizer.json";
if (modelName.ToLower().Contains("world") || modelName.ToLower().Contains("finch") || modelName.ToLower().Contains("rwkv-6"))
    tokenizerFileName = "rwkv_vocab_v20230424.txt";
rf.RegisterRWKVGGMLModel(Path.Combine(path, modelName), Path.Combine(path, tokenizerFileName));

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