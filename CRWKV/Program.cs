using RWKV;

Console.Write("Input Model Name(rwkv-4-pile-169m-uint8.onnx): ");
var modelName = Console.ReadLine();
if (string.IsNullOrEmpty(modelName))
    modelName = "rwkv-4-pile-169m-uint8.onnx";

Console.Write("ctx_len(1024): ");
var ctx_len = 1024;
var ctx_len_str = Console.ReadLine();
if (!string.IsNullOrEmpty(ctx_len_str))
    ctx_len = int.Parse(ctx_len_str);

Console.Write("n_layer(12): ");
var n_layer = 12;
var n_layer_str = Console.ReadLine();
if (!string.IsNullOrEmpty(n_layer_str))
    n_layer = int.Parse(n_layer_str);

Console.Write("n_embd(768): ");
var n_embd = 768;
var n_embd_str = Console.ReadLine();
if (!string.IsNullOrEmpty(n_embd_str))
    n_embd = int.Parse(n_embd_str);

Console.WriteLine($"Loading({modelName})[{ctx_len},{n_layer},{n_embd}]...");
var r = new Runner(modelName, ctx_len, n_layer, n_embd);
r.Init();
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