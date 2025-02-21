using RWKVSharp.Core;
using RWKVSharp.Core.Sampler;
using RWKVSharp.Core.Tokenizer;
using System.Runtime.InteropServices;

namespace RWKVSharp
{
    public static class RunnerFactoryExtend
    {
        private static bool _isLoadLibrary = false;

        public static void LoadLibrary()
        {
            if (_isLoadLibrary)
                return;

            _isLoadLibrary = true;
            var library_name = RwkvCppNative.LIBRARY_NAME;
            var library_ex = "";
            var os = "";

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                os = "win";
                library_ex = ".dll";
            }
            else
            {
                library_name = $"lib{library_name}";
            }

            //优先加载目录下的
            var libraryPath = $"{library_name}{library_ex}";
            if (File.Exists(libraryPath))
            {
                NativeLibrary.Load(libraryPath);
                return;
            }

            libraryPath = $"{RwkvCppNative.LIBRARY_NAME}{library_ex}";
            if (File.Exists(libraryPath))
            {
                NativeLibrary.Load(libraryPath);
                return;
            }

            //再根据系统和CPU架构加载
            var arch = RuntimeInformation.ProcessArchitecture switch
            {
                Architecture.X64 => "x64",
                Architecture.X86 => "x86",
                _ => ""
            };

            var avx = "";
            if (RuntimeInformation.ProcessArchitecture == Architecture.X86 || RuntimeInformation.ProcessArchitecture == Architecture.X64)
            {
                if (
                    System.Runtime.Intrinsics.X86.Avx512BW.IsSupported ||
                    System.Runtime.Intrinsics.X86.Avx512CD.IsSupported ||
                    System.Runtime.Intrinsics.X86.Avx512DQ.IsSupported ||
                    System.Runtime.Intrinsics.X86.Avx512F.IsSupported ||
                    System.Runtime.Intrinsics.X86.Avx512Vbmi.IsSupported)
                {
                    avx = "avx512";
                }
                else if (System.Runtime.Intrinsics.X86.Avx2.IsSupported)
                {
                    avx = "avx2";
                }
                else if (System.Runtime.Intrinsics.X86.Avx.IsSupported)
                {
                    avx = "avx";
                }
            }

            if (!string.IsNullOrWhiteSpace(os) && !string.IsNullOrWhiteSpace(arch) && !string.IsNullOrWhiteSpace(avx))
            {
                libraryPath = Path.Combine("runtimes", $"{os}-{arch}", "native", avx, $"{library_name}{library_ex}");
                if (File.Exists(libraryPath))
                {
                    NativeLibrary.Load(libraryPath);
                    return;
                }
                libraryPath = Path.Combine("runtimes", $"{os}-{arch}", "native", avx, $"{RwkvCppNative.LIBRARY_NAME}{library_ex}");
                if (File.Exists(libraryPath))
                {
                    NativeLibrary.Load(libraryPath);
                    return;
                }
            }
        }

        public static void RegisterRWKVGGMLModel(this RunnerFactory runnerFactory, string modelPath, string tokenizerPath, uint? n_gpu_layers = null)
        {
            LoadLibrary();
            runnerFactory.RegisterRWKVGGMLModel("Default", modelPath, tokenizerPath, n_gpu_layers);
        }

        public static void RegisterRWKVGGMLModel(this RunnerFactory runnerFactory, string name, string modelPath, string tokenizerPath, uint? n_gpu_layers = null)
        {
            LoadLibrary();
            runnerFactory.RegisterRunner<RwkvRunner>(
                name,
                new RwkvModel(modelPath, n_gpu_layers: n_gpu_layers),
                new RunnerOptions()
                {
                    Tokenizer = () =>
                    {
                        var name = Path.GetFileNameWithoutExtension(tokenizerPath);
                        return name switch
                        {
                            "rwkv_vocab_v20230424" => new TrieTokenizer(tokenizerPath),
                            _ => new BPETokenizer(tokenizerPath)
                        };
                    },
                    Sampler = () =>
                    {
                        return new RwkvSampler();
                    }
                }
            );
        }
    }
}
