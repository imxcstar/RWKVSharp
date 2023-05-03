namespace RWKV
{
    public static class RunnerFactoryExtend
    {
        public static void SetGGMLModel(this RunnerFactory runnerFactory)
        {
            runnerFactory.Model = new GGMLModel(Path.Combine(runnerFactory.ModelPath, runnerFactory.ModelFile));
        }
    }
}
