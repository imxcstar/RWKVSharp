using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CLLM.Core.Interfaces
{
    public interface IRunner
    {
        public string Name { get; }
        public IModel? Model { get; }
        public IRunnerOptions? Options { get; set; }

        public void Init(string name, IModel? model, IRunnerOptions? options);

        public void Run(string value, Action<string> callBack, object? rawValue = null);
        public string Run(string value, object? rawValue = null);
        public IAsyncEnumerable<string> RunAsync(string value, object? rawValue = null);
    }
}
