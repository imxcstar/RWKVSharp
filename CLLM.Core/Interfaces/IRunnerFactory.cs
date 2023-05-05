using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CLLM.Core.Interfaces
{
    public interface IRunnerFactory
    {
        public void RegisterRunner<T>(string name, IModel? model, IRunnerOptions? runnerOptions) where T : IRunner, new();
        public IRunner Builder(string name);

        public void RegisterRunner<T>(IModel? model, IRunnerOptions? runnerOptions) where T : IRunner, new();
        public IRunner Builder();
    }
}
