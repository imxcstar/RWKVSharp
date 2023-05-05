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

        public virtual void Run(string value, Action<string> callBack, object? rawValue = null)
        {
            var task = Task.Run(async () =>
            {
                await foreach (var item in RunAsync(value, rawValue))
                {
                    callBack?.Invoke(item);
                }
            });
            task.Wait();
        }

        public virtual string Run(string value, object? rawValue = null)
        {
            var ret = new List<string>();
            var task = Task.Run(async () =>
            {
                await foreach (var item in RunAsync(value, rawValue))
                {
                    ret.Add(item);
                }
            });
            task.Wait();
            return string.Join("", ret);
        }

        public abstract IAsyncEnumerable<string> RunAsync(string value, object? rawValue = null);
    }
}
