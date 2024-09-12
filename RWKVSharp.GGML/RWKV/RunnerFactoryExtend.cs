using RWKVSharp.Core;
using RWKVSharp.Core.Sampler;
using RWKVSharp.Core.Tokenizer;
using RWKVSharp.Core.Tokenizer.RWKV;

namespace RWKVSharp
{
    public static class RunnerFactoryExtend
    {
        public static void RegisterRWKVGGMLModel(this RunnerFactory runnerFactory, string modelPath, string tokenizerPath)
        {
            runnerFactory.RegisterRWKVGGMLModel("Default", modelPath, tokenizerPath);
        }

        public static void RegisterRWKVGGMLModel(this RunnerFactory runnerFactory, string name, string modelPath, string tokenizerPath)
        {
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
