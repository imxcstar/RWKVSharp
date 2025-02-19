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
            var library_ex = "";
            var os = "";

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                os = "win";
                library_ex = ".dll";
            }

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
                var libraryPath = Path.Combine("runtimes", $"{os}-{arch}", "native", avx, $"{RwkvCppNative.LIBRARY_NAME}{library_ex}");
                if (File.Exists(libraryPath))
                    NativeLibrary.Load(libraryPath);
            }
        }

        public static void RegisterRWKVGGMLModel(this RunnerFactory runnerFactory, string modelPath, string tokenizerPath)
        {
            LoadLibrary();
            runnerFactory.RegisterRWKVGGMLModel("Default", modelPath, tokenizerPath);
        }

        public static void RegisterRWKVGGMLModel(this RunnerFactory runnerFactory, string name, string modelPath, string tokenizerPath)
        {
            LoadLibrary();
            runnerFactory.RegisterRunner<RwkvRunner>(
                name,
                new RwkvModel(modelPath),
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
