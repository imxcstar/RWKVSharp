using RWKVSharp.Core.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RWKVSharp.Core
{
    public class RunnerBuilder<T> : IRunnerFactoryBuilder where T : IRunner, new()
    {
        private string _name;
        private IModel _model;
        private IRunnerOptions _options;

        public RunnerBuilder(string name, IModel model, IRunnerOptions options)
        {
            _name = name;
            _model = model;
            _options = options;
        }

        public IRunner Builder()
        {
            var runner = new T();
            runner.Init(_name, _model, _options);
            return runner;
        }
    }
}
