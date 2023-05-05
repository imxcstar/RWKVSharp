using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CLLM.Core.Interfaces;

namespace CLLM.Core
{
    public class RunnerOptions : IRunnerOptions
    {
        public ITokenizer? Tokenizer { get; set; }
        public int MaxTokens { get; set; } = 128;
    }
}
