using System.IO;

namespace RWKV
{
    public static class RunnerFactoryExtend
    {
        public static void SetOnnxModel(this RunnerFactory runnerFactory)
        {
            runnerFactory.Model = new OnnxModel(Path.Combine(runnerFactory.ModelPath, runnerFactory.ModelFile), runnerFactory.N_embd, runnerFactory.N_layer);
        }
    }
}
