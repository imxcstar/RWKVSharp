using CLLM.Core;
using RWKV;

Console.Write("Input Model Name(rwkv-169m-ggml-f16.bin): ");
var modelName = Console.ReadLine();
if (string.IsNullOrEmpty(modelName))
    modelName = "rwkv-169m-ggml-f16.bin";

Console.WriteLine($"Loading...");

var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Model");
var rf = new RunnerFactory();
rf.RegisterRWKVGGMLModel(Path.Combine(path, modelName), Path.Combine(path, "20B_tokenizer.json"));

var r = rf.Builder();

while (true)
{
    Console.Write(">");
    var value = Console.ReadLine();
    if (string.IsNullOrEmpty(value))
        continue;
    r.Run(value.Replace("\\r\\n", "\r\n").Replace("\\r", "\r").Replace("\\n", "\n"), Console.Write);
    Console.WriteLine();
}