using CLLM.Core.Sampler;
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

        public void InitInstruction(string instruction);
        public void Run(string value, Action<string> callBack, RunOptions? options = null);
        public string Run(string value, RunOptions? options = null);
        public IAsyncEnumerable<string> RunAsync(string value, RunOptions? options = null);
    }

    public class RunOptions
    {
        public int MaxTokens { get; set; } = 512;
        public ISampler? Sampler { get; set; }
    }
}
