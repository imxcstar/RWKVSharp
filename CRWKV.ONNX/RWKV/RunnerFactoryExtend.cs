using CLLM.Core;
using CLLM.Core.Sampler;
using CLLM.Core.Tokenizer;
using CLLM.Core.Tokenizer.RWKV;
using System.IO;

namespace RWKV
{
    public static class RunnerFactoryExtend
    {
        public static void RegisterRWKVOnnxModel(this RunnerFactory runnerFactory, string modelPath, string tokenizerPath, int embed, int layers)
        {
            runnerFactory.RegisterRWKVOnnxModel("Default", modelPath, tokenizerPath, embed, layers);
        }

        public static void RegisterRWKVOnnxModel(this RunnerFactory runnerFactory, string name, string modelPath, string tokenizerPath, int embed, int layers)
        {
            runnerFactory.RegisterRunner<Runner>(
                name,
                new OnnxModel(modelPath, embed, layers),
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
