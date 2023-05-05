using CLLM.Core;
using RWKV;

Console.Write("Input Model Name(RWKV_32_2560_16.onnx): ");
var modelName = Console.ReadLine();
if (string.IsNullOrEmpty(modelName))
    modelName = "RWKV_32_2560_16.onnx";

var modelNames = modelName.Split("_");
var n_layer = int.Parse(modelNames[1]);
var n_embd = int.Parse(modelNames[2]);

Console.WriteLine($"Loading...");

var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Model");
var rf = new RunnerFactory();
rf.RegisterRWKVOnnxModel(Path.Combine(path, modelName), Path.Combine(path, "20B_tokenizer.json"), n_embd, n_layer);

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