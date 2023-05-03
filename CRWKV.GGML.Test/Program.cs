using RWKV;

Console.Write("Input Model Name(rwkv-169m-ggml-f16.bin): ");
var modelName = Console.ReadLine();
if (string.IsNullOrEmpty(modelName))
    modelName = "rwkv-169m-ggml-f16.bin";

Console.WriteLine($"Loading...");

var rf = new RunnerFactory(modelName);
rf.Init();
rf.SetGGMLModel();

var r = rf.NewRunner();

while (true)
{
    Console.Write(">");
    var value = Console.ReadLine();
    if (string.IsNullOrEmpty(value))
        continue;
    r.Run(value, Console.Write);
    Console.WriteLine();
}