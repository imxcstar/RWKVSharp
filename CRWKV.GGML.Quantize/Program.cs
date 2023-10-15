using RWKV;

Console.Write("Input GGML Model File Path: ");
var modelFilePath = Console.ReadLine();
if (string.IsNullOrEmpty(modelFilePath) || !File.Exists(modelFilePath))
    throw new Exception($"\"{modelFilePath}\" does not exist!");

Console.Write("Input Quantize Format Name(Q4_0/Q4_1/Q5_0/Q5_1/Q8_0): ");
var formatName = Console.ReadLine()?.ToUpper();
if (string.IsNullOrEmpty(formatName) || !"Q4_0/Q4_1/Q5_0/Q5_1/Q8_0".Contains(formatName))
    throw new Exception($"\"{formatName}\" format error, the format should be one of (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0)");

Console.Write("Output Model Path(Optional): ");
var outPath = Console.ReadLine();
if (string.IsNullOrEmpty(outPath))
    outPath = Path.Combine(Path.GetDirectoryName(modelFilePath) ?? "", $"{Path.GetFileNameWithoutExtension(modelFilePath)}-{formatName}.bin");

Console.WriteLine($"On the way...");
if (RwkvCppNative.rwkv_quantize_model_file(modelFilePath, outPath, formatName))
    Console.WriteLine($"Finish.");
else
    Console.WriteLine($"Fail!");