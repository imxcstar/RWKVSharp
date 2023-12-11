using CLLM.Core.Interfaces;
using CLLM.Core.Sampler;
using CLLM.Core.Tokenizer;
using System.Collections.Generic;

namespace CLLM.Core
{
    public class Runner : RunnerBase
    {
        private ITokenizer _tokenizer;
        private ISampler _sampler;
        private object _state;

        public override void Init(string name, IModel model, IRunnerOptions options)
        {
            base.Init(name, model, options);
            _state = model.GetEmptyStates();
            _tokenizer = options.Tokenizer.Invoke();
            _sampler = options.Sampler.Invoke();
        }

        public override void InitInstruction(string instruction)
        {
            var tokens = _tokenizer.Encode(instruction);
            _state = Model.GetStates(tokens.ToArray());
        }

        public override async IAsyncEnumerable<string> RunAsync(string value, object? rawValue = null)
        {
            var xutput = new Queue<int>(_tokenizer.Encode(value));
            var size = xutput.Count;
            var input = 0;
            var strEnc = new List<int>();
            for (int i = 0; i < size + Options.MaxTokens; i++)
            {
                if (xutput.Count > 0)
                {
                    input = xutput.Dequeue();
                }
                var logits_state = await Task.FromResult(Model.Forward(input, _state));
                _state = logits_state.state;

                if (xutput.Count == 0)
                {
                    var ac = await Task.FromResult(_sampler.Sample(logits_state.logits));
                    if (ac == 0)
                        break;
                    input = ac;
                    strEnc.Add(ac);
                    var str = _tokenizer.Decode(strEnc);
                    if (!str.Contains("\ufffd"))
                    {
                        yield return str;
                        strEnc.Clear();
                    }
                }
            }
        }
    }
}
