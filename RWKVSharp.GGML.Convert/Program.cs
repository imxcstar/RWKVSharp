using Python.Included;
using Python.Runtime;
using System.IO;
using System.Text;

Installer.InstallPath = AppContext.BaseDirectory;
Installer.LogMessage += s =>
{
    Console.WriteLine(s);
};
await Installer.SetupPython();
await Installer.TryInstallPip();
await Installer.PipInstallModule("torch");
await Installer.PipInstallModule("torchvision");
await Installer.PipInstallModule("torchaudio");

PythonEngine.Initialize();
dynamic sys = Py.Import("sys");
sys.path.append(Path.Combine(AppContext.BaseDirectory, "Python"));

dynamic RWKVConverter = Py.Import("RWKVConverter");

Console.Write("Input PyTorch Model File Path: ");
var modelFilePath = Console.ReadLine()?.Trim('"');
if (string.IsNullOrEmpty(modelFilePath) || !File.Exists(modelFilePath))
    throw new Exception($"\"{modelFilePath}\" does not exist!");

Console.Write("Input Data Type Name(FP16/FP32)[FP16]: ");
var dataType = Console.ReadLine()?.ToUpper();
if (string.IsNullOrEmpty(dataType))
    dataType = "FP16";
if (!"FP16/FP32".Contains(dataType))
    throw new Exception($"\"{dataType}\" type error, the type should be one of (FP16/FP32)");

Console.Write("Output GGML Model Path(Optional): ");
var outPath = Console.ReadLine();
if (string.IsNullOrEmpty(outPath))
    outPath = Path.Combine(Path.GetDirectoryName(modelFilePath) ?? "", $"{Path.GetFileNameWithoutExtension(modelFilePath)}-{dataType}.bin");
else
    outPath = Path.Combine(outPath, $"{Path.GetFileNameWithoutExtension(modelFilePath)}-{dataType}.bin");

var converter = RWKVConverter.RWKVConverter(modelFilePath, outPath, dataType);
converter.convert();