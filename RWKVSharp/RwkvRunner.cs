using RWKVSharp.Core;
using RWKVSharp.Core.Interfaces;
using RWKVSharp.Core.Sampler;
using RWKVSharp.Core.Tokenizer;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace RWKVSharp
{
    public class RwkvRunner : RunnerBase
    {
        private ITokenizer _tokenizer = null!;
        private ISampler _sampler = null!;
        private object _state = null!;

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
            _state = Model!.GetStates([.. tokens]);
        }

        public override async IAsyncEnumerable<string> GenerateAsync(string value, RunOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var tokens = _tokenizer.Encode(value);
            var input = 0;
            var strEnc = new List<int>();
            var maxTokens = options?.MaxTokens ?? 512;
            var sampler = options?.Sampler ?? _sampler;
            (IEnumerable<float> logits, object state) logits_state;

            for (int i = 0; i < maxTokens + 1; i++)
            {
                if (cancellationToken.IsCancellationRequested)
                    break;

                if (i == 0)
                {
                    logits_state = await Task.FromResult(Model!.Forward(tokens, _state));
                }
                else
                {
                    logits_state = await Task.FromResult(Model!.Forward(input, _state));
                }
                _state = logits_state.state;

                var ac = await Task.FromResult(sampler.Sample(logits_state.logits));
                if (ac == 0)
                    break;
                input = ac;
                strEnc.Add(ac);
                var str = _tokenizer.Decode(strEnc);
                if (!str.Contains('\ufffd'))
                {
                    yield return str;
                    strEnc.Clear();
                }
            }
        }
    }
}
