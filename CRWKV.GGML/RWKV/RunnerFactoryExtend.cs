using CLLM.Core;

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
                    Tokenizer = new Tokenizer(tokenizerPath)
                }
            );
        }
    }
}
