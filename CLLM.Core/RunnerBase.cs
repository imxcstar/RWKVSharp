using CLLM.Core.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CLLM.Core
{
    public abstract class RunnerBase : IRunner
    {
        private string _name;
        public virtual string Name => _name;

        private IModel? _model;
        public virtual IModel? Model => _model;

        public virtual IRunnerOptions? Options { get; set; }

        public virtual void Init(string name, IModel? model, IRunnerOptions? options)
        {
            _name = name;
            _model = model;
            Options = options;
        }

        public abstract void InitInstruction(string instruction);

        public virtual void Run(string value, Action<string> callBack, RunOptions? options = null)
        {
            var task = Task.Run(async () =>
            {
                await foreach (var item in RunAsync(value, options))
                {
                    callBack?.Invoke(item);
                }
            });
            task.Wait();
        }

        public virtual string Run(string value, RunOptions? options = null)
        {
            var ret = new List<string>();
            var task = Task.Run(async () =>
            {
                await foreach (var item in RunAsync(value, options))
                {
                    ret.Add(item);
                }
            });
            task.Wait();
            return string.Join("", ret);
        }

        public abstract IAsyncEnumerable<string> RunAsync(string value, RunOptions? options = null);
    }
}
