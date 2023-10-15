using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CLLM.Core.Interfaces;
using CLLM.Core.Sampler;
using CLLM.Core.Tokenizer;

namespace CLLM.Core
{
    public class RunnerOptions : IRunnerOptions
    {
        public Func<ITokenizer> Tokenizer { get; set; }
        public Func<ISampler> Sampler { get; set; }
        public int MaxTokens { get; set; } = 128;
    }
}
