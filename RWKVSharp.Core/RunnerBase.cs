using RWKVSharp.Core.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RWKVSharp.Core
{
    public abstract class RunnerBase : IRunner
    {
        private string _name = null!;
        public virtual string Name => _name;

        private IModel _model = null!;
        public virtual IModel Model => _model;

        public virtual IRunnerOptions Options { get; set; } = null!;

        public virtual void Init(string name, IModel model, IRunnerOptions options)
        {
            _name = name;
            _model = model;
            Options = options;
        }

        public abstract void InitInstruction(string instruction);

        public virtual void Generate(string value, Action<string> callBack, RunOptions? options = null)
        {
            var task = Task.Run(async () =>
            {
                await foreach (var item in GenerateAsync(value, options))
                {
                    callBack?.Invoke(item);
                }
            });
            task.Wait();
        }

        public virtual string Generate(string value, RunOptions? options = null)
        {
            var ret = new List<string>();
            var task = Task.Run(async () =>
            {
                await foreach (var item in GenerateAsync(value, options))
                {
                    ret.Add(item);
                }
            });
            task.Wait();
            return string.Join("", ret);
        }

        public abstract IAsyncEnumerable<string> GenerateAsync(string value, RunOptions? options = null, CancellationToken cancellationToken = default);
    }
}
