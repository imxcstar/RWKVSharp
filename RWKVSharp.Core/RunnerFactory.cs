using RWKVSharp.Core.Interfaces;
using System.Xml.Linq;

namespace RWKVSharp.Core
{
    public class RunnerFactory : IRunnerFactory
    {
        private Dictionary<string, IRunnerFactoryBuilder> _runnerRegistry;

        public RunnerFactory()
        {
            _runnerRegistry = new Dictionary<string, IRunnerFactoryBuilder>();
        }

        public IRunner Builder(string name)
        {
            return _runnerRegistry[name].Builder();
        }

        public IRunner Builder()
        {
            return Builder("Default");
        }

        public void RegisterRunner<T>(string name, IModel model, IRunnerOptions runnerOptions) where T : IRunner, new()
        {
            _runnerRegistry.Add(name, new RunnerBuilder<T>(name, model, runnerOptions));

        }

        public void RegisterRunner<T>(IModel model, IRunnerOptions runnerOptions) where T : IRunner, new()
        {
            RegisterRunner<T>("Default", model, runnerOptions);
        }
    }
}
