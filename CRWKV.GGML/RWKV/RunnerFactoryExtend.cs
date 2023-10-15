using CLLM.Core;
using CLLM.Core.Sampler;
using CLLM.Core.Tokenizer;
using CLLM.Core.Tokenizer.RWKV;

namespace RWKV
{
    public static class RunnerFactoryExtend
    {
        public static void RegisterRWKVGGMLModel(this RunnerFactory runnerFactory, string modelPath, string tokenizerPath)
        {
            runnerFactory.RegisterRWKVGGMLModel("Default", modelPath, tokenizerPath);
        }

        public static void RegisterRWKVGGMLModel(this RunnerFactory runnerFactory, string name, string modelPath, string tokenizerPath)
        {
            runnerFactory.RegisterRunner<Runner>(
                name,
                new GGMLModel(modelPath),
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
                        return new NPSampler();
                    }
                }
            );
        }
    }
}
