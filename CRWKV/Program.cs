using RWKV;

Console.Write("Input Model Name(RWKV_32_2560_16.onnx): ");
var modelName = Console.ReadLine();
if (string.IsNullOrEmpty(modelName))
    modelName = "RWKV_32_2560_16.onnx";

var modelNames = modelName.Split("_");
var n_layer = int.Parse(modelNames[1]);
var n_embd = int.Parse(modelNames[2]);

Console.WriteLine($"Loading...");

var rf = new RunnerFactory(modelName, n_layer, n_embd);
rf.Init();
var r = rf.NewRunner();

while (true)
{
    Console.Write(">");
    var value = Console.ReadLine();
    if (string.IsNullOrEmpty(value))
        continue;
    r.Run(value, v =>
    {
        Console.Write(v);
    });
    Console.WriteLine();
}